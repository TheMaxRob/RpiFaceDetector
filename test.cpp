#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

Mat detectAndAlignFace(Mat& image, CascadeClassifier& faceDetector) {
    vector<Rect> faces;
    Mat gray;
    
    if (image.channels() > 1) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    faceDetector.detectMultiScale(
        gray,          
        faces,         
        1.05,          
        2,             
        0,             
        Size(20, 20)   
    );

    // Save detected face regions for debugging purposes
    Mat debugImage = image.clone();
    for (const auto& face : faces) {
        rectangle(debugImage, face, Scalar(255, 0, 0), 2);
    }
    static int imageCount = 0;
    imwrite("debug_detection_" + to_string(imageCount++) + ".jpg", debugImage);
    
    if (faces.empty()) {
        return Mat();
    }
    
    Rect* largestFace = &faces[0];
    for (auto& face : faces) {
        if (face.area() > largestFace->area()) {
            largestFace = &face;
        }
    }
    
    Mat faceROI = gray(*largestFace);
    Mat resizedFace;
    resize(faceROI, resizedFace, Size(200, 200));
    
    Mat equalizedFace;
    equalizeHist(resizedFace, equalizedFace);
    
    return equalizedFace;
}


int main() {

    // Load detector
    CascadeClassifier faceDetector;
    if (!faceDetector.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        cerr << "Error loading face detection model" << endl;
        return -1;
    }

    // Load Recognition Model
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    try {
        recognizer->read("face_model.yml");
    } catch (const cv::Exception& e) {
        cerr << "Error loading recognition model: " << e.what() << endl;
        return -1;
    }

    // Load and process
    string testImagePath = "test_charlie.jpg";
    Mat testImage = imread(testImagePath);
    if (testImage.empty()) {
        cerr << "Could not load test image: " << testImagePath << endl;
        return -1;
    }


    // Detect & Align Face
    Mat processedFace = detectAndAlignFace(testImage, faceDetector);
    if (processedFace.empty()) {
        cerr << "Failed to process face in image " << endl;
        return -1;
    }


    // Prediction
    int predictedLabel = -1;
    double confidence = 0.0;
    try {
        recognizer->predict(processedFace, predictedLabel, confidence);
    } catch (const cv::Exception& e) {
        cout << "Error predicting image: " << e.what() << endl;
        return -1;
    }

    const double CONFIDENCE_THRESHOLD = 80.0;
    cout << "Prediction Results: " << predictedLabel << endl;
    cout << "Confidence: " << confidence << endl;

    if (confidence >= CONFIDENCE_THRESHOLD) {
        cout << "Confidence too low to recognize face.";
    } else {
        if (predictedLabel == 1) {
            cout << "Prediction is: Sophia" << " with confidence " << confidence << endl;
        } 
        else if (predictedLabel == 2) {
            cout << "Prediction is: Charlie" <<  " with confidence " << confidence << endl;
        } 
    }
    
    return 0;
}