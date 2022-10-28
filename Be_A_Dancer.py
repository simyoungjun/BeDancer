#%%
from msilib.schema import Error
import os

import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import pytube
from ffpyplayer.player import MediaPlayer
from multiprocessing import Process, Value, Array,freeze_support
import threading
import os
#%%
class BeDancer():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_style = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    def __init__(self, const_k=0.6):
        self.__const_k = const_k
        self.__video_download_path = "video"
        self.__keypoints_path = "keypoints"
        self.__dance_name = None
        self.__accumulate_acc = []
        self.speed = 1
        self.isMirr = True
        
    def set_const_k(self):
        self.__const_k = float(input("난이도 조절(0~1 사이 값): "))
    def get_const_k(self):
        print(f"현재 난이도: {self.__const_k}")
    def __get_accumlate_acc(self):
        return self.__accumulate_acc
    def __save_dance_name(self):
        self.__dance_name = input("누구의 무슨 춤? ex) 안유진 러브다이브 : ")
    def set_dance_name(self, s):
        self.__dance_name = s
        
    def print_dance_data(self):
        acc_acc = self.__get_accumlate_acc()
        accMax, accMin, accMean = np.max(acc_acc), np.min(acc_acc), np.mean(acc_acc)
        print(f"Max Acc: {accMax}\tMin Acc: {accMin}\tAvg. Acc: {accMean}\n")
        acc_acc = pd.DataFrame(acc_acc)
        acc_acc.plot(figsize=(25, 6))
        plt.title("Accuarcy for Frames")
        plt.xlabel("Frames")
        plt.ylabel("Accuarcy")
        plt.legend("Acc")
        plt.axhline(y=70, color="r")
        plt.show()
        
    def download_video(self):
        self.__save_dance_name()
        url = input(f"{self.__dance_name}의 안무 영상 링크: ")
        if not os.path.exists(self.__video_download_path): os.mkdir(self.__video_download_path)
        yt = pytube.YouTube(url).streams.filter(res="720p").first()
        yt.download(output_path=self.__video_download_path, filename=self.__dance_name+".mp4")
        

    
    def __draw_skeleton(self, image, skeleton):
        
        # 오른쪽 스켈레톤 (붉은색)
        cv2.line(image, skeleton[12], skeleton[14], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/어깨 -> 오/팔꿈치
        cv2.line(image, skeleton[14], skeleton[16], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/팔꿈치 -> 오/손목
        cv2.line(image, skeleton[12], skeleton[24], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/어깨 -> 오/엉덩이
        cv2.line(image, skeleton[24], skeleton[26], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/엉덩이 -> 오/무릎
        cv2.line(image, skeleton[26], skeleton[28], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/무릎 -> 오/발목
        cv2.line(image, skeleton[28], skeleton[30], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/발목 -> 오/뒷꿈치
        cv2.line(image, skeleton[30], skeleton[32], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오른발
        cv2.line(image, skeleton[28], skeleton[32], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오른발
        # 왼쪽 스켈레톤 (푸른색)
        cv2.line(image, skeleton[11], skeleton[13], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/어깨 -> 왼/팔꿈치
        cv2.line(image, skeleton[13], skeleton[15], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/팔꿈치 -> 왼/손목
        cv2.line(image, skeleton[11], skeleton[23], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/어깨 -> 왼/엉덩이
        cv2.line(image, skeleton[23], skeleton[25], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/엉덩이 -> 왼/무릎
        cv2.line(image, skeleton[25], skeleton[27], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/무릎 -> 왼/발목
        cv2.line(image, skeleton[27], skeleton[29], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/발목 -> 왼/뒷꿈치
        cv2.line(image, skeleton[29], skeleton[31], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼발
        cv2.line(image, skeleton[27], skeleton[31], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼발
        # 상체 스켈레톤 (회색)
        cv2.line(image, skeleton[11], skeleton[12], (224, 224, 224), thickness=5, lineType=cv2.LINE_AA, shift=None)
        cv2.line(image, skeleton[23], skeleton[24], (224, 224, 224), thickness=5, lineType=cv2.LINE_AA, shift=None)   
        
    
    def __load_cor_data(self):
        with open(self.__keypoints_path+"/"+self.__dance_name+"_keypoints.json", "r") as keypoints:
            data = json.load(keypoints)
            return np.array(data)
        
    def extract_keypoints(self, isMirr=True, showExtract=False):
        if not os.path.exists(self.__keypoints_path): os.mkdir(self.__keypoints_path)
        
        keypoints_list = []
        self.isMirr = isMirr
        cv2.startWindowThread()
        cap = cv2.VideoCapture(os.path.join(self.__video_download_path, self.__dance_name+".mp4")) #저장한 영상 가져옴
        
        with self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret: break
                if isMirr == False:
                    dance_image = cv2.flip(image, 1)
                
                results = pose.process(image)
                # Extracting
                try: 
                    keypoints_list.append([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark])
                except: pass
                if showExtract:
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(244, 244, 244), thickness=2, circle_radius=1),
                                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(153, 255, 153), thickness=2, circle_radius=1))
                    cv2.imshow("Extracting", image)
                    if cv2.waitKey(1)==ord("q"): break
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
        # Save coord. Data for json type
        with open(self.__keypoints_path+"/"+self.__dance_name+"_keypoints.json", "w") as keypoints:
            json.dump(keypoints_list, keypoints)
            

    def norm2(self,vector):
        return np.sqrt(np.sum(vector**2, axis=1))
    
    def unit_vector(self, landmarks):
        x = [12, 14, 12, 24, 26, 28, 30, 28, 11, 13, 11, 23, 25, 27, 29, 27, 11, 23]
        y = [14, 16, 24, 26, 28, 30, 32, 32, 13, 15, 23, 25, 27, 29, 31, 31, 12, 24]
        landmarks_vector = landmarks[x] - landmarks[y]
        landmarks_unit_vector = landmarks_vector / np.sqrt(np.sum(landmarks_vector**2, axis=1))[:,np.newaxis]
        return landmarks_unit_vector
    
    def normalized_to_coordinate(self, normalized_pose, image_shape):
        coordinate_pose = normalized_pose[:, :2] * np.array([image_shape[1], image_shape[0]])  # 프레임 별 pose x,y 좌표* [width,height] 만큼 scaling
        coordinate_pose = np.asarray(coordinate_pose, dtype=int)   # pose 따라 line 그리기 위해 float -> int로 변환
        return  coordinate_pose
    
    def scoring(self,dancer_unit_vectors,user_unit_vectors):
        accuracy = int(np.sum(np.abs(self.unit_vector(dancer_unit_vectors) - self.unit_vector(user_unit_vectors))))
    
    def crop_image_width(self,image, width_after_crop):
        image_width = image.shape[1]
        width_after_crop=560
        if image_width>width_after_crop:
            mid = image_width//2
            return image[: , mid - width_after_crop // 2 : mid + width_after_crop // 2]
        else:
            return image
        
    def accuracy_thread(self):
        global continue_window
        global frame_num
        global dance_image
        global user_image
        global dance_pose
        global accuracy
        global lock
        
        while frame_num < 0:
            pass
        
        print(frame_num)
        
        cv2.startWindowThread()
        
        while continue_window == True :
            
            lock.acquire()
            with self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                dance_image_thread = dance_image.copy()
                user_image_thread = user_image.copy()
                dance_landmarks_thread= dance_pose.copy()
                lock.release()
                
                user_results = pose.process(user_image_thread)
                
                
                try:
                    user_landmarks_thread=np.array([[lmk.x, lmk.y, lmk.z] for lmk in user_results.pose_landmarks.landmark])
                
                
                    #error 계산할때 사용되는 image와 dance pose
                    coordinate_dance_pose = self.normalized_to_coordinate(dance_landmarks_thread, dance_image_thread.shape)
                    self.__draw_skeleton(dance_image_thread, coordinate_dance_pose) # user image에 dance pose 그림
                    
                    coordinate_user_pose =self.normalized_to_coordinate(user_landmarks_thread, user_image_thread.shape)
                    self.__draw_skeleton(user_image_thread, coordinate_user_pose)
                    
                    thread_dancerANDuser_with_landmarks =np.hstack((dance_image_thread, user_image_thread))
                    cv2.imshow("Extracting", thread_dancerANDuser_with_landmarks)
                    
                    if cv2.waitKey(1)==ord("q"): break
                
                
                
                    # 추출해 온 데이터
                    # print('사용자 pose 추출')
                    norm2_of_unit_vectors = self.norm2(self.unit_vector(dance_landmarks_thread) - self.unit_vector(user_landmarks_thread))
                    accuracy = 100 - int(np.sum(norm2_of_unit_vectors) / (2 * norm2_of_unit_vectors.shape[0]) * 100)
                    # accuracy = int(np.sum(np.abs(self.unit_vector(dance_landmarks_thread) - self.unit_vector(user_landmarks_thread))))

                    # print('thread에서 계산한 frame_num,  accuracy : ', frame_num, ' ', accuracy)
                except: pass  

    
    def play_dance(self):
        #전역변수 선언
        global continue_window
        global frame_num
        global dance_image
        global user_image
        global dance_pose
        global accuracy
        global lock
        accuracy=0
        
        
        lock = threading.Lock()
        
        # cv2.startWindowThread()
        dance = cv2.VideoCapture(os.path.join(self.__video_download_path, self.__dance_name+".mp4"))
        # user = cv2.VideoCapture('C:/Users/sim/Downloads/Just-DDance-main/video/제니 솔로.mp4')
        user = cv2.VideoCapture(os.path.join(self.__video_download_path,"러브다이브.mp4"))
        # try: user = cv2.VideoCapture(0)
        # except: user = cv2.VideoCapture(1)

        user.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        user.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        player = MediaPlayer(os.path.join(self.__video_download_path, self.__dance_name+".mp4"))
        
        dance_poses = self.__load_cor_data() #안무영상 pose 상대좌표 (전역변수 초기화)
        # skeletons = {}
        FPS = dance.get(cv2.CAP_PROP_FPS) # 안무영상(dance) frame rate
        pose_point = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] #joint index
        
        frame_num = -1
        acc_per_frame = 0
        continue_window = user.isOpened()
        # with self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        #Thread 실행
        th = threading.Thread(target = self.accuracy_thread, name = 'thread')
        th.start()
        
        prev_time = time.time() # 출력 FPS 30으로 맞추기 위한 변수 (이전 frame 출력시간)
        while continue_window == True:

            lock.acquire() 
            frame_num += 1
            dance_ret, dance_image = dance.read()
            cropped_dance_image = self.crop_image_width(dance_image,560)

            user_ret, user_image = user.read()
            if self.isMirr ==False:
                dance_image = cv2.flip(dance_image, 1)
            user_image = cv2.flip(user_image, 1)
            
            try:
                dance_pose = dance_poses[frame_num]
            except: break
            
            if not user_ret: break
            if not dance_ret: break
            lock.release()


            coordinate_dance_pose = self.normalized_to_coordinate(dance_pose,dance_image.shape) + [(user_image.shape[1] - dance_image.shape[1]) // 2, 0] # normalized 포즈 좌표로 변환후 user image의 중앙에 배치
            # coordinate_dance_pose = dance_pose[:, :2] * np.array([dance_image.shape[1], dance_image.shape[0]]) + (user_image.shape[1] - dance_image.shape[1])/2# 프레임 별 pose x,y 좌표* [width,height] 만큼 scaling
            # coordinate_dance_pose = np.asarray(coordinate_dance_pose, dtype = int) # pose 따라 line 그리기 위해 float -> int로 변환

            user_image_drawed = user_image.copy()
            self.__draw_skeleton(user_image_drawed, coordinate_dance_pose) # user image에 dance pose 그림
            
            while(time.time()-prev_time < float(bd.speed)/FPS):
                pass
                
            audio_frame, val = player.get_frame()
            # h_output = np.hstack((cv2.flip(dance_image, 1), user_image))
            # user_image = cv2.cvtColor(cv2.flip(user_image, 1), cv2.COLOR_BGR2RGB)
            cv2.putText(user_image_drawed, 'Score : '+str(accuracy), (20, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            
            output = np.hstack((cropped_dance_image, user_image_drawed))
            
            cv2.imshow("Be a Dancer!", output)
            prev_time = time.time()
            
            
            if cv2.waitKey(1)==ord("q"):
                break 
            
        continue_window = False
        th.join()
        player.close_player()
        user.release() 
        dance.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        

# global go
# global frame_num
# global dance_image
# global user_image
# global dance_pose
# global accuracy
    
if __name__=='__main__':
    # freeze_support()
    bd = BeDancer()
    # isDownload = input("영상을 다운 받아야하나요? (Y/N): ")
    # if isDownload.upper()=="Y":
    #     bd.download_video()
    #     ism = input("영상이 거울 모드인가요? (Y/N): ")
    #     isk = input("키포인트 출력 과정을 보고 싶나요? (Y/N): ")
    #     if ism.upper()=="Y": ism=True
    #     else: ism=False
    #     if isk.upper()=="Y": isk=True
    #     else: isk=False
    #     print("키 포인트 추출이 좀 오래걸립니다! (영상길이+a)")
    #     bd.extract_keypoints(ism, isk)
    # else:
    #     n = input("다운 받은 영상의 이름이 뭔가요?: ")
    #     bd.set_dance_name(n)        
    #     isKeypoint = input("키포인트를 추출했었나요? (Y/N): ")
    #     if isKeypoint.upper()=="Y":
    #         pass

    #     else: 
    #         isk = input("키포인트 출력 과정을 보고 싶나요? (Y/N): ")
    #         print("키 포인트 추출이 좀 오래걸립니다! (영상길이+a)")
    #         bd.extract_keypoints(isMirr=True, showExtract=True)
    # print("실행 중입니다! 잠시만 기다려주세요! (실행 중 q를 누르면 종료됩니다)")
    
    # bd.speed = input("몇 배속으로 하시겠습니까? (type: float): ")
    bd.set_dance_name('솔로')        
    bd.play_dance()
    # jd.print_dance_data()
    



# %