# Dance Practice System Using Deep Learning Pose Estimation

---

### **Overview**
This project implements a dance practice system using deep learning-based pose estimation algorithms and multithreading techniques. By leveraging MediaPipe and real-time pose estimation, the system allows users to download choreography videos, visualize keypoints of dancers, and calculate pose similarity scores in real time. The multithreading approach ensures smooth video playback and efficient performance, even on resource-constrained hardware.

---

### **Features**
1. **Choreography Crawling**:
   - Easily download choreography videos from YouTube using the Pytube library.
2. **Pose Estimation**:
   - Uses MediaPipe’s BlazePose model to extract keypoints from dancers in choreography videos.
3. **Real-Time Scoring**:
   - Calculates similarity scores between the user and the dancer using cosine similarity of pose vectors.
4. **Multithreading for Performance**:
   - Separate threads for output rendering and similarity scoring ensure smooth operation.

---

### **System Flow**
| **System Flowchart** |
|-----------------------|
| ![System Flowchart](image.png) |

1. Download choreography videos using crawling.
2. Extract keypoints for each frame using BlazePose.
3. Capture the user’s video via webcam.
4. Calculate pose similarity scores in real time.
5. Display results, including choreography, user video, pose guide, and similarity scores.

---

### **Key Technologies**
- **MediaPipe BlazePose**: A deep learning-based pose estimation framework for detecting human keypoints.
- **Multithreading**: Implements parallel processing for output rendering and similarity scoring.
- **Cosine Similarity**: Calculates pose similarity scores, invariant to scale and position changes.

| **Threading Workflow** |
|-------------------------|
| ![Threading Workflow](image.png) |

---

### **How It Works**
1. **Keypoint Extraction**:
   - Keypoints from the dancer are visualized as pose guides to help users mimic choreography.
2. **Real-Time Scoring**:
   - User pose vectors are compared with choreographer vectors to calculate similarity.
3. **Multithreading Design**:
   - **Output Thread**: Renders videos, pose guides, and similarity scores at high FPS.
   - **Scoring Thread**: Performs real-time pose estimation and scoring asynchronously.

---

### **Results**
1. **Accuracy**:
   - Achieved 100% similarity scores in controlled scenarios where user and dancer poses are identical.
2. **Performance**:
   - On Intel Core i7-12700, scoring takes ~0.27 seconds per frame. Performance is robust even on lower-end CPUs.
3. **Pose Guide Visualization**:
   - Real-time feedback enables effective dance practice.

---

### **Future Directions**
- Expand to support multiple users simultaneously.
- Integrate feedback for pose correction.
- Enhance pose estimation robustness for dynamic environments.

---

