#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <fstream>
#include <sstream>
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

    faceDetector.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
    
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

void readCSV(const string& filename, vector<Mat>& images, vector<int>& labels, CascadeClassifier& faceDetector) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open CSV file." << endl;
        exit(1);
    }

    string line, path, label;
    int skipped = 0;
    int processed = 0;

    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, ';');
        getline(liness, label);
        
        if (!path.empty() && !label.empty()) {
            Mat image = imread(path);
            if (image.empty()) {
                cerr << "Error: Could not read image at path: " << path << endl;
                skipped++;
                continue;
            }

            // Detect and align face
            Mat processedFace = detectAndAlignFace(image, faceDetector);
            if (processedFace.empty()) {
                cerr << "Warning: No face detected in " << path << endl;
                skipped++;
                continue;
            }

            images.push_back(processedFace);
            labels.push_back(stoi(label));
            processed++;
            
            cout << "Processed: " << path << " (Label: " << label << ")" << endl;
        }
    }
    
    cout << "Training data summary:" << endl;
    cout << "Successfully processed: " << processed << " images" << endl;
    cout << "Skipped: " << skipped << " images" << endl;
}

int main() {
    // Load face detector
    CascadeClassifier faceDetector;
    if (!faceDetector.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        cerr << "Error loading face detection model" << endl;
        return -1;
    }

    vector<Mat> images;
    vector<int> labels;
    string csvFile = "friend_faces.csv";

    readCSV(csvFile, images, labels, faceDetector);

    if (images.empty()) {
        cerr << "No valid training images found" << endl;
        return -1;
    }

    // Create & train recognizer
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    recognizer->train(images, labels);

    // Save model
    recognizer->save("face_model.yml");
    cout << "Training complete; model saved to face_model.yml" << endl;
    
    return 0;
}