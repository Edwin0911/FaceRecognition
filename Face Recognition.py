# -*- coding: utf-8 -*-
import os,dlib,glob,numpy,cv2,time
from skimage import io

def add_user():
    # key insert
    user_name = input("Enter your name:")
    
    #路徑請自行更新
    img_file = "C:/Users/edwin/Documents/Face Recognition/rec/"
    
    #選擇第一隻攝影機
    cap = cv2.VideoCapture( 0)
    
    #取得預設的臉部偵測器
    detector = dlib.get_frontal_face_detector()
    
    i=0
    print("Please look at the camera...")
    while(cap.isOpened()):
        #讀出frame資訊
        ret, frame = cap.read()

        #偵測人臉
        dets = detector(frame, 1) 
        
        
        if ( len(dets) == 1 ):
          
          
          cv2.imwrite( img_file + user_name + str(i) +'.jpg', frame)  
          print("Save image...")
          i+=1
          time.sleep(1)
          
        if ( i == 5): 
            print("Add user successful!")
            break
def delete_user():
    # key insert
    
    #路徑請自行更新
    img_file = "C:/Users/edwin/Documents/Face Recognition/rec/"
    
    while(True):       
        user_name = input("Enter a name:")  
        try:
            i = 0
            for i in range(5):
                delete_img = img_file + user_name + str(i) + ".jpg"
                print( "Delete image:" + delete_img )
                os.remove( img_file + user_name + str(i) + ".jpg" )
            
        except:
            print("No user exist")
            return
        
        print("User delete")
        return
def detect_user():
    #選擇第一隻攝影機
    cap = cv2.VideoCapture( 0)

    # 人臉68特徵點模型路徑
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    # 人臉辨識模型路徑
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
    
    # 比對人臉圖片資料夾名稱
    faces_folder_path = "./rec"
       
    #取得預設的臉部偵測器
    detector = dlib.get_frontal_face_detector()
    #detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    #根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
    sp = dlib.shape_predictor( predictor_path )
    
    # 載入人臉辨識檢測器
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    
    # 比對人臉描述子列表
    descriptors = []
    
    # 比對人臉名稱列表
    candidate = []
    
    # 針對比對資料夾裡每張圖片做比對:
    # 1.人臉偵測
    # 2.特徵點偵測
    # 3.取得描述子
    #print( glob.glob(os.path.join(faces_folder_path, "*.jpg") )
    i = 0      
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
      base = os.path.basename(f)
      # 依序取得圖片檔案人名
      img = io.imread(f)
    
      # 1.人臉偵測
      
      dets = detector(img, 1)
      if ( len(dets) == 1 ):
        #加入圖片檔案人名至列表
        i+=1
        candidate.append(os.path.splitext(base)[ 0])
        for k, d in enumerate(dets):
          # 2.特徵點偵測
          shape = sp(img, d)
          
          # 3.取得描述子，128維特徵向量
          face_descriptor = facerec.compute_face_descriptor(img, shape)
          print(face_descriptor)
          # 轉換numpy array格式
          v = numpy.array(face_descriptor)
          descriptors.append(v)
    if ( i == 0): 
        print("No user exist!")
        return      
              
    #當攝影機打開時，對每個frame進行偵測
    while(cap.isOpened()):
        #讀出frame資訊
        ret, frame = cap.read()
        
        cv2.imwrite( 'before.jpg', frame) 
        
        a = 2
        O = frame * float(a)
        O[O > 255] = 255
        O = numpy.round(O)
        O = O.astype(numpy.uint8)
        frame = O
        
        cv2.imwrite( 'after.jpg', frame) 

        #偵測人臉
        dets = detector(frame, 1) 
        
        if ( len(dets) == 1 ):

            dist = []
            for k, d in enumerate(dets):
              shape = sp(frame, d)
              face_descriptor = facerec.compute_face_descriptor(frame, shape)
              d_test = numpy.array(face_descriptor)
            
              x1 = d.left()
              y1 = d.top()
              x2 = d.right()
              y2 = d.bottom()
              # 以方框標示偵測的人臉
              cv2.rectangle(frame, (x1, y1), (x2, y2), ( 255, 255, 255), 4, cv2. LINE_AA)
              
             
            # 計算歐式距離
            for i in descriptors:
              dist_ = numpy.linalg.norm(i -d_test)
              dist.append(dist_)
        
            # 將比對人名和比對出來的歐式距離組成一個dict
            c_d = dict( zip(candidate,dist))
            
            # 根據歐式距離由小到大排序
            cd_sorted = sorted(c_d.items(), key = lambda d:d[ 1])
            # 取得最短距離就為辨識出的人名
            
            #以相似程度判斷
            if ( cd_sorted[ 0][ 1] < 0.4 ):
                rec_name = "Correct"        
                cv2.putText(frame, rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 0, 255, 0), 2, cv2. LINE_AA)
                
            else:
                rec_name = "Error"
                cv2.putText(frame, rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 0, 0, 255), 2, cv2. LINE_AA)
            # 將辨識出的人名印到圖片上面

        #輸出到畫面
        cv2.imshow( "Face Detection(press Esc to option)", frame)
    
        #如果按下ESC键，就退出
        if cv2.waitKey( 10) == 27:
           break
        
    #釋放記憶體
    cap.release()
    #關閉所有視窗
    cv2.destroyAllWindows()
def main():
    while(True):
        detect_user()
        option = int(input("Enter 1 to add user:\nEnter 2 to delete user:\nEnter 0 to exit:\nEnter 3 to continue:\nInput your choice:"))        
        if(option == 1):
            add_user()
        elif(option == 2):
            delete_user()
        elif(option == 3):
            detect_user()
        elif(option == 0):
            print("Program Exit...")
            time.sleep(3)
            return

main()