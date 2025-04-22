import numpy as np
import multiprocessing
import cv2
import struct
from typing import Tuple, List
import time

# Constants from paper
NUMBER_OF_PROCESSES = 3  # Mỗi kênh màu một tiến trình
CONFUSION_DIFFUSION_ROUNDS = 5
CONFUSION_SEED_UPPER_BOUND = 10000
CONFUSION_SEED_LOWER_BOUND = 3000
PRE_ITERATIONS = 1000
BYTES_RESERVED = 6
PI = np.arccos(-1)

# Cat map parameters
P = 3
Q = 11

class PLCM:
    def __init__(self, control_param: float, init_condition: float):
        if control_param >= 0.5:
            control_param = 1 - control_param
        self.p = control_param
        self.x = init_condition
        self._pre_iterate()
    
    def _pre_iterate(self):
        for _ in range(PRE_ITERATIONS):
            self.x = self._single_iterate(self.x)
    
    def _single_iterate(self, x: float) -> float:
        if 0 <= x < self.p:
            return x / self.p
        elif self.p <= x <= 0.5:
            return (x - self.p) / (0.5 - self.p)
        else:
            return self._single_iterate(1.0 - x)
    
    def _double_to_bytes(self, value: float) -> bytes:
        return struct.pack('d', value)[2:2+BYTES_RESERVED]
    
    def iterate_and_get_bytes(self, iterations: int) -> Tuple[float, List[bytes]]:
        x = self.x
        byte_list = []
        
        for _ in range(iterations):
            x = self._single_iterate(x)
            byte_list.append(self._double_to_bytes(x))
            
        self.x = x
        return x, byte_list

class ImageEncryptionSystem:
    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.uint8)
        self.height, self.width = image.shape[0], image.shape[1]
        
        self.main_plcm = PLCM(control_param=0.37, init_condition=0.2)
        
        self.temp_frame = np.zeros_like(image)
        self.confused_frame = np.zeros_like(image)
        self.diffused_frame = np.zeros_like(image)
        self.diffusion_seed = 0
    
    def _generate_plcm_params(self):
        x = self.main_plcm.x
        
        p1 = x
        x = self.main_plcm._single_iterate(x)
        x1 = x
        x = self.main_plcm._single_iterate(x)
        p2 = x
        x = self.main_plcm._single_iterate(x)
        x2 = x
        
        return (p1, x1), (p2, x2)
    
    def confusion_operation(self):
        transformationMatrix = np.array([[1,P], [Q, P*Q +1]])
        
        for r in range(self.height):
            for c in range(self.width):
                
                old_pos = np.array([[r],[c]])
                new_pos = np.dot(transformationMatrix,old_pos) 
                self.confused_frame[new_pos[0][0]% self.height, new_pos[1][0]%self.width] = self.temp_frame[r, c]
                    
    def generate_byte_sequence(self) -> np.ndarray:
        pixels = self.height * self.width
        iterations = (pixels + BYTES_RESERVED - 1) // BYTES_RESERVED
        
        plcm1 = PLCM(*self._generate_plcm_params()[0])
        plcm2 = PLCM(*self._generate_plcm_params()[1])
        
        _, bytes1 = plcm1.iterate_and_get_bytes(iterations)
        _, bytes2 = plcm2.iterate_and_get_bytes(iterations)
        
        arr1 = np.frombuffer(b''.join(bytes1), dtype=np.uint8)
        arr2 = np.frombuffer(b''.join(bytes2), dtype=np.uint8)
        return np.bitwise_xor(arr1, arr2)[:pixels]
                    
    def diffusion_operation(self):
        byte_seq = self.generate_byte_sequence()
        
        seq_idx = 0
        for i in range(self.height):
            for j in range(self.width):
                byte = byte_seq[seq_idx]
                
                if i == 0 and j == 0:
                    temp_sum = (self.temp_frame[i, j].astype(np.int32) + byte) % 256
                    self.diffused_frame[i, j] = byte ^ temp_sum ^ self.diffusion_seed
                else:
                    prev_i = i if j > 0 else i-1
                    prev_j = j-1 if j > 0 else self.width-1
                    
                    prev_pixel = self.diffused_frame[prev_i, prev_j]
                    temp_sum = (self.temp_frame[i, j].astype(np.int32) + byte) % 256
                    self.diffused_frame[i, j] = byte ^ temp_sum ^ prev_pixel
                
                seq_idx += 1
                
    def encrypt(self) -> np.ndarray:
        current_frame = self.image.copy()
        
        for _ in range(CONFUSION_DIFFUSION_ROUNDS):
            # Reset PLCM
            self.main_plcm = PLCM(0.37, 0.2)
            
            # Confusion phase
            self.confused_frame = np.zeros_like(current_frame)
            self.temp_frame = current_frame.copy()
            self.confusion_operation()
            
            # Update current frame after confusion
            current_frame = self.confused_frame.copy()
            
            # Generate diffusion seed and prepare for diffusion
            x = self.main_plcm._single_iterate(self.main_plcm.x)
            self.diffusion_seed = int(x * 256) & 0xFF
            self.diffused_frame = np.zeros_like(current_frame)
            self.temp_frame = current_frame.copy()
            
            # Diffusion phase
            self.diffusion_operation()
            
            # Update current frame after diffusion
            current_frame = self.diffused_frame.copy()
        
        return current_frame

def encrypt_channel(channel, queue, channel_name):
    system = ImageEncryptionSystem(channel)
    encrypted = system.encrypt()
    queue.put((channel_name, encrypted))  # Đưa kết quả vào queue

def encrypt_image_BGR(image):
    B, G, R = cv2.split(image)
    
    # Tạo queue để nhận kết quả từ các process
    queue = multiprocessing.Queue()
    
    # Tạo và chạy các process
    process_B = multiprocessing.Process(target=encrypt_channel, args=(B, queue,'B'))
    process_G = multiprocessing.Process(target=encrypt_channel, args=(G, queue, 'G'))
    process_R = multiprocessing.Process(target=encrypt_channel, args=(R, queue, 'R'))
    
    process_B.start()
    process_G.start()
    process_R.start()
    
    # Lấy kết quả từ queue
    results = []
    for _ in range(3):
        result = queue.get()
        results.append(result)
    
    process_B.join()
    process_G.join()
    process_R.join()
    
    # Sắp xếp kết quả theo thứ tự BGR
    results.sort(key=lambda x: x[0])
    
    # Gộp các kênh màu đã mã hóa
    encrypted_B = results[0][1]
    encrypted_G = results[1][1]
    encrypted_R = results[2][1]

    # Gộp các kênh màu đã mã hóa
    encrypted_image_color = cv2.merge((encrypted_B, encrypted_G, encrypted_R))
    return encrypted_image_color
def encrypted_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap:
        print("Could not open video")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    #Tao 
    outputfile = 'encryptedVideo.mp4'
    encryptedVideo = cv2.VideoWriter(outputfile, fourcc,fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        encrypted_frame = encrypt_image_BGR(frame)
        encryptedVideo.write(encrypted_frame)
        '''cv2.imshow('frame',frame)
        cv2.imshow('encryptedframe',encrypted_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
    cap.release()
    encryptedVideo.release()
    cv2.destroyAllWindows()
    return outputfile

def open_Video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video:
        print("Could not open video")
        exit()
    while True:
        ret,frame =video.read()
        if not ret:
            break
        if ret == True:
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
               break

    video.release()
    cv2.destroyAllWindows()
def main():
    '''image_path = 'cameraman_resize.png'
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    
    start_time = time.time()
    encrypted_image_color = encrypt_image_BGR(image)
    end_time = time.time()
    
    process_time = end_time - start_time

    cv2.imshow('encrypted', encrypted_image_color)
    cv2.imwrite('cameraman_resize_encrypted.png', encrypted_image_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Thời gian xử lý: {process_time:.4f} giây")
    print(encrypted_image_color.shape)'''
    
    video_path ='output_video.mp4'
    start_time = time.time()
    encryptedVideo_path = encrypted_video(video_path)
    open_Video(encryptedVideo_path)
    end_time =time.time()
    process_time = end_time-start_time
    print(process_time)
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
