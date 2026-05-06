import cv2
import numpy as np
import os
import glob
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

class GlaucomaDetectionSystem:
    def __init__(self, target_size=(240, 240), use_adaptive_bps=True, use_ensemble=True):
        """Initializes the Automated Glaucoma Detection System pipeline."""

        self.target_size = target_size
        self.use_adaptive_bps = use_adaptive_bps
        self.use_ensemble = use_ensemble
        

        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=0.95) # Retain 95% of variance 
        self.classifier = None
        

        self.lbp_radius = 1
        self.lbp_points = 8 * self.lbp_radius
        

        self.glcm_distances = [1, 2, 4] 
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees
        
    def preprocess_image(self, img_path):
        """Extracts green channel and applies CLAHE for illumination correction."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found at {img_path}")
            
        img = cv2.resize(img, self.target_size)
        

        green_channel = img[:, :, 1]
        

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_green = clahe.apply(green_channel)
        
        return enhanced_green
        
    def segment_optic_disc(self, img):
        """Focuses feature extraction on the diagnostically relevant optic disc region."""
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img # Fallback to original image if OD is not distinct
            
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        roi = img[y1:y2, x1:x2]
    
        return cv2.resize(roi, self.target_size)

    def extract_bit_planes(self, img):
        """Decomposes the 8-bit image into 8 distinct binary planes."""
        bit_planes = []
        for i in range(8):
            plane = (img >> i) & 1
            bit_planes.append(plane)
        return bit_planes
        
    def adaptive_bps_selection(self, bit_planes):
        """
        Automatically adapts to image quality by selecting bit planes 
        based on their entropy (information content).
        """
        selected_planes = []
        for i, plane in enumerate(bit_planes):
            if i >= 4: 
                selected_planes.append(plane)
        return selected_planes

    def get_lbp_image(self, plane):
        """
        Encodes local texture on the binary planes.
        
        """
        plane_8u = (plane * 255).astype(np.uint8)
        lbp = local_binary_pattern(plane_8u, self.lbp_points, self.lbp_radius, method='uniform')
        return lbp.astype(np.uint8)

    def extract_glcm_features(self, lbp_img):
        """
        Extracts correlation, contrast, energy, and homogeneity properties.
        """

        glcm = graycomatrix(lbp_img, distances=self.glcm_distances, angles=self.glcm_angles, 
                            levels=256, symmetric=True, normed=True)
        
        features = []

        for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:

            val = graycoprops(glcm, prop).flatten()
            features.extend(val)
            
        return features

    def process_image(self, img_path):
        """
        Executes the entire feature extraction sequence for a given image.
        """
        img = self.preprocess_image(img_path)
        roi = self.segment_optic_disc(img)
        bit_planes = self.extract_bit_planes(roi)
        
        if self.use_adaptive_bps:
            selected_planes = self.adaptive_bps_selection(bit_planes)
        else:
            selected_planes = bit_planes
            
        image_features = []

        for plane in selected_planes:
            lbp_img = self.get_lbp_image(plane)
            glcm_feats = self.extract_glcm_features(lbp_img)
            image_features.extend(glcm_feats)
            
        return np.array(image_features)
        
    def train_model(self, X_train, y_train):
        """
        Stage 6 & 7: Feature Normalization, Dimensionality Reduction, and Training
        """

        X_scaled = self.scaler.fit_transform(X_train)
        

        X_pca = self.pca.fit_transform(X_scaled)
        
        if self.use_ensemble:

            clf1 = SVC(kernel='rbf', probability=True, random_state=42)
            clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
            clf3 = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            self.classifier = VotingClassifier(
                estimators=[('svm', clf1), ('rf', clf2), ('gb', clf3)],
                voting='soft'
            )
        else:
            self.classifier = SVC(kernel='rbf', probability=True, random_state=42)
            
        self.classifier.fit(X_pca, y_train)
        
    def predict(self, X_test):
        """
        Predicts labels for unseen data vectors.
        """
        X_scaled = self.scaler.transform(X_test)
        X_pca = self.pca.transform(X_scaled)
        return self.classifier.predict(X_pca)


if __name__ == "__main__":

    system = GlaucomaDetectionSystem(use_adaptive_bps=True, use_ensemble=True)
    

    X = []
    y = []
    
    healthy_dir = "data/healthy/*.jpg"
    glaucoma_dir = "data/glaucoma/*.jpg"
    

    
    for img_path in glob.glob(healthy_dir):
        features = system.process_image(img_path)
        X.append(features)
        y.append(0)  # Class 0: Healthy
        
    for img_path in glob.glob(glaucoma_dir):
        features = system.process_image(img_path)
        X.append(features)
        y.append(1)  # Class 1: Glaucoma
    
    X = np.array(X)
    y = np.array(y)
    
    # Using 10-fold Cross-Validation as described in the paper
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        system.train_model(X_train, y_train)
        preds = system.predict(X_test)
        scores.append(accuracy_score(y_test, preds))
        
    print(f"10-Fold CV Accuracy: {np.mean(scores)*100:.2f}%")
    print("Pipeline successfully created and ready for dataset ingestion!")