#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

// Function to detect and align face
Mat detectAndAlignFace(Mat& image, CascadeClassifier& faceDetector) {
    vector<Rect> faces;
    Mat gray;
    
    // Grayscale
    if (image.channels() > 1) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Detect faces
    faceDetector.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
    
    if (faces.empty()) {
        cerr << "No face detected in the image" << endl;
        return Mat();
    }
    
    // Assume largest detected face is correct face
    Rect* largestFace = &faces[0];
    for (auto& face : faces) {
        if (face.area() > largestFace->area()) {
            largestFace = &face;
        }
    }
    
    // Resize face region
    Mat faceROI = gray(*largestFace);
    Mat resizedFace;
    resize(faceROI, resizedFace, Size(200, 200));
    
    // Apply histogram equalization to normalize lighting
    Mat equalizedFace;
    equalizeHist(resizedFace, equalizedFace);
    
    return equalizedFace;
}

int main() {
    // Load face detector
    CascadeClassifier faceDetector;
    if (!faceDetector.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        cerr << "Error loading face detection model" << endl;
        return -1;
    }

    // Load recognition model
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    try {
        recognizer->read("face_model.yml");
    } catch (const cv::Exception& e) {
        cerr << "Error loading recognition model: " << e.what() << endl;
        return -1;
    }

    // Load & process test image
    string testImagePath = "test_charlie.jpg";
    Mat testImage = imread(testImagePath);
    if (testImage.empty()) {
        cerr << "Could not load test image: " << testImagePath << endl;
        return -1;
    }

    // Detect and align face
    Mat processedFace = detectAndAlignFace(testImage, faceDetector);
    if (processedFace.empty()) {
        cerr << "Failed to process face in the image" << endl;
        return -1;
    }

    // // Save processed face for debugging
    // imwrite("processed_face.jpg", processedFace);
    // cout << "Saved processed face image for verification" << endl;

    // Predict
    int predictedLabel = -1;
    double confidence = 0.0;
    try {
        recognizer->predict(processedFace, predictedLabel, confidence);
    } catch (const cv::Exception& e) {
        cerr << "Error during prediction: " << e.what() << endl;
        return -1;
    }

    const double CONFIDENCE_THRESHOLD = 80.0;
    
    cout << "Prediction Results:" << endl;
    cout << "Confidence Score: " << confidence << endl;
    
    if (confidence >= CONFIDENCE_THRESHOLD) {
        cout << "No confident match found (confidence score too high)" << endl;
    } else {
        if (predictedLabel == 1) {
            cout << "Predicted: Sophia (Confidence: " << confidence << ")" << endl;
        } else if (predictedLabel == 2) {
            cout << "Predicted: Charlie (Confidence: " << confidence << ")" << endl;
        } else {
            cout << "Unknown label: " << predictedLabel << endl;
        }
    }

    return 0;
}