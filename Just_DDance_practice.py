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
class JustDDance():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_style = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    def __init__(self, const_k=0.6):
        self.__const_k = const_k
        self.__video_download_path = "video"
        self.__keypoints_path = "keypoints"
        self.__dance_name = None
        self.__accumulate_acc = []
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
        
    def __get_margin(self, user_tri, dance_tri):
        margin = []
        ut = [(user_tri[0][0]+user_tri[1][0]+user_tri[2][0])/3, (user_tri[0][1]+user_tri[1][1]+user_tri[2][1])/3, (user_tri[0][2]+user_tri[1][2]+user_tri[2][2])/3]
        dt = [(dance_tri[0][0]+dance_tri[1][0]+dance_tri[2][0])/3, (dance_tri[0][1]+dance_tri[1][1]+dance_tri[2][1])/3, (dance_tri[0][2]+dance_tri[1][2]+dance_tri[2][2])/3]
        for u, d in zip(ut, dt): margin.append(u-d)
        return margin
    
    def __get_distance(self, pt1, pt2):
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
    
    def __load_cor_data(self):
        with open(self.__keypoints_path+"/"+self.__dance_name+"_keypoints.json", "r") as keypoints:
            data = json.load(keypoints)
            return np.array(data)
        
    def extract_keypoints(self, isMirr=False, showExtract=False):
        if not os.path.exists(self.__keypoints_path): os.mkdir(self.__keypoints_path)
        
        keypoints_list = []
        cv2.startWindowThread()
        cap = cv2.VideoCapture(os.path.join(self.__video_download_path, self.__dance_name+".mp4")) #저장한 영상 가져옴
        
        with self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret: break
                if not isMirr: image = cv2.flip(image, 1)
                
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
            
    def error(self,user_image,dance_cors,pose_point,frame_num,acc):
        print('자식 프로세스 pid : ',os.getpid())
        print('자식 프로세스 frame num : ',frame_num.value)
        
        acc_per_frame = []
        user_image = cv2.cvtColor(cv2.flip(user_image, 1), cv2.COLOR_BGR2RGB)
        with self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            user_results = pose.process(user_image)
            user_image = cv2.cvtColor(user_image, cv2.COLOR_RGB2BGR)
            # 사용자
            # self.mp_drawing.draw_landmarks(user_image, user_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            #                             landmark_drawing_spec = self.mp_drawing.DrawingSpec(color=(244, 244, 244), thickness=2, circle_radius=1),
            #                             connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(153, 255, 153), thickness=2, circle_radius=1))
            try:
                # user_input = {str(idx): [lmk.x, lmk.y, lmk.z] for idx, lmk in enumerate(user_results.pose_landmarks.landmark)}
                user_input=np.array([[lmk.x, lmk.y, lmk.z] for lmk in user_results.pose_landmarks.landmark])
            except: pass
            # 추출해 온 데이터
            try:
                print('사용자 pose 추출')
                # get coors MARGIN
                cors_margin = self.__get_margin([user_input[0], user_input[23], user_input[24]], [dance_cors[frame_num.value][0], dance_cors[frame_num.value][23], dance_cors[frame_num.value][24]])
                # pose_point = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                # x_cor_pose, y_cor_pose, z_cor_pose = int((dance_cors[frame_num.value][pose_point][0]+cors_margin[0])*user_image.shape[1]), int((dance_cors[frame_num.value][pose_point][1]+cors_margin[1])*user_image.shape[0]), int((dance_cors[frame_num][pose_point][2]+cors_margin[2])*1000)
                # cv2.circle(user_image, (x_cor_pose, y_cor_pose), 8, (244, 244, 244), cv2.FILLED)
                # L2 Norm
                # acc_per_frame.append(np.round(self.__const_k / (np.linalg.norm([(x_cor_pose/user_image.shape[1]-cors_margin[0])-user_input[str(pose_point)][0], (y_cor_pose/user_image.shape[0]-cors_margin[1])-user_input[str(pose_point)][1], (z_cor_pose/1000-cors_margin[2])-user_input[str(pose_point)][2]]) + self.__const_k), 2))
                
                
                print('np.mean(acc_per_frame)*100 : ',np.mean(cors_margin)*100)
                acc.value = np.mean(cors_margin)*100
                # self.__accumulate_acc.append(acc)
                
                print('자식 acc.value : ', acc.value)
            except: pass       
            
    def stack_userImg_with_pose(self, dance_image, user_image, dance_cors) :
        coordinate_dance_pose = dance_cors[frame_num.value,:,:2]*np.array([dance_image.shape[1],dance_image.shape[1]])# 프레임 별 pose x,y 좌표* [width,height] 만큼 scaling
        coordinate_dance_pose = np.asarray(coordinate_dance_pose, dtype = int) # pose 따라 line 그리기 위해 float -> int로 변환
        # skeletons[pose_point] = (x_cor_pose, y_cor_pose)
        self.__draw_skeleton(user_image, coordinate_dance_pose)
        
        while(1):
            if current_time>1./FPS:
                audio_frame, val = player.get_frame()
                h_output = np.hstack((cv2.flip(dance_image, 1), user_image))
                cv2.imshow("Just DDance!", h_output)
                prev_time = time.time()
                break
            else:
                current_time = time.time()-prev_time
                print(current_time)
        
        if cv2.waitKey(1)==ord("q"):
            break     
    
    def BeDancer(self):

        
        cv2.startWindowThread()
        dance = cv2.VideoCapture(os.path.join(self.__video_download_path, self.__dance_name+".mp4"))
        # user = cv2.VideoCapture('C:/Users/sim/Downloads/Just-DDance-main/video/제니 솔로.mp4')
        user = cv2.VideoCapture(os.path.join(self.__video_download_path, self.__dance_name+".mp4"))
        # try: user = cv2.VideoCapture(0)
        # except: user = cv2.VideoCapture(1)
        
        user.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        user.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        player = MediaPlayer(os.path.join(self.__video_download_path, self.__dance_name+".mp4"))
        
        Dance_pose = self.__load_cor_data() #안무영상 pose 상대좌표
        # skeletons = {}
        prev_time = 0 # 출력 FPS 30으로 맞추기 위한 변수
        FPS = dance.get(cv2.CAP_PROP_FPS) # 안무영상(dance) frame rate
        pose_point = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] #joint index
        frame_num = Value('i',-1)
        acc = Value('d',0)
        child_process_end = Value('')
        # with self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        process_list=[]
        
        while user.isOpened():
            frame_num.value+=1

            current_time = time.time()-prev_time
            user_ret, user_image = user.read()
            dance_ret, dance_image = dance.read()
            if not user_ret: break
            if not dance_ret: break
    
            
                        #멀티 프로세스 분기
            process = Process(target=self.error, args=[user_image,dance_cors, pose_point, frame_num, acc])
            print('부모 프로세스 pid : ', os.getpid())
            # process_list.append(process)
            if (os.getpid())
            process.start()
            # process.join()
            print('부모에서 frame num : ', frame_num.value)
            print('부모에서 acc : ', acc.value)
            print('자식 return 값 : ', process)
            print('\n')
             
        for process in process_list:
            process.join()
            print(process.is_alive())
            
        cv2.putText(user_image, str(acc.value)+"%", (20, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        
        # process1.join()   
        player.close_player()
        user.release() 
        dance.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        


    
if __name__=='__main__':
    # freeze_support()
    jd = JustDDance()
    # isDownload = input("영상을 다운 받아야하나요? (Y/N): ")
    # if isDownload.upper()=="Y":
    #     jd.download_video()
    #     ism = input("영상이 거울 모드인가요? (Y/N): ")
    #     isk = input("키포인트 출력 과정을 보고 싶나요? (Y/N): ")
    #     if ism.upper()=="Y": ism=True
    #     else: ism=False
    #     if isk.upper()=="Y": isk=True
    #     else: isk=False
    #     print("키 포인트 추출이 좀 오래걸립니다! (영상길이+a)")
    #     jd.extract_keypoints(ism, isk)
    # else:
    #     n = input("다운 받은 영상의 이름이 뭔가요?: ")
    #     jd.set_dance_name(n)        
    #     isKeypoint = input("키포인트를 추출했었나요? (Y/N): ")
    #     if isKeypoint.upper()=="Y":
    #         pass
    #     else: 
    #         isk = input("키포인트 출력 과정을 보고 싶나요? (Y/N): ")
    #         print("키 포인트 추출이 좀 오래걸립니다! (영상길이+a)")
    #         jd.extract_keypoints(isMirr=True, showExtract=True)
    # print("실행 중입니다! 잠시만 기다려주세요! (실행 중 q를 누르면 종료됩니다)")
    
    
    jd.set_dance_name('안유진 러브다이브')        
    
    jd.BeDancer()
    # jd.print_dance_data()
    



# %
