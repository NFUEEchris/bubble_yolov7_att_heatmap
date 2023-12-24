#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw,ImageFont

from yolo import YOLO, YOLO_ONNX

import os 
import shutil
import statistics
import matplotlib.pyplot as plt
def split_image(input_image_path, output_directory, rows, cols):
            image = Image.open(input_image_path)
            
            # 获取图像的宽度和高度
            width, height = image.size
            
            # 计算每个子图像的宽度和高度
            tile_width = width // cols
            tile_height = height // rows
            
            # 确保输出目录存在
            os.makedirs(output_directory, exist_ok=True)
            
            # 循环遍历每个子图像
            for i in range(rows):
                for j in range(cols):
                    left = j * tile_width
                    upper = i * tile_height
                    right = left + tile_width
                    lower = upper + tile_height
                    
                    # 切割子图像
                    tile = image.crop((left, upper, right, lower))
                    
                    # 保存子图像
                    output_path = os.path.join(output_directory, f'tile_{i}_{j}.jpg')
                    tile.save(output_path)
def merge_images(input_directory, output_image_path, rows, cols):
    images = []
    for i in range(rows):
        row_images = []
        for j in range(cols):
            image_path = os.path.join(input_directory, f'tile_{i}_{j}.jpg')
            tile = Image.open(image_path)
            row_images.append(tile)
        images.append(row_images)

    widths, heights = zip(*(i.size for i in images[0]))
    total_width = sum(widths)
    total_height = sum(heights)

    merged_image = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for row_images in images:
        x_offset = 0
        for tile in row_images:
            merged_image.paste(tile, (x_offset, y_offset))
            x_offset += tile.width
        y_offset += row_images[0].height

    merged_image.save(output_image_path)

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             在影片中計數統計，最高、最低、中位數、眾數。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在yolo.py_423行左右处的YOLO_ONNX
    #   'merge_predict'     將圖片切成 m*n 等分要設定 m_rows 和 m_cols
    #   'Grad_CAM_predict'  熱力圖預測輸出
    #----------------------------------------------------------------------------------------------------------#
    mode = "Grad_CAM_predict"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = True
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = "img\\716441414.362062.mp4"
    video_save_path = "img_out\\716441414.362062.mp4"
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "C:\\code\\yolov7-pytorch-master\\VOCdevkit\\VOC2007\\JPEGImages\\vannamei18"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    m_rows = 2  # 行数
    m_cols = 1  # 列数

    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    # if mode != "predict_onnx":
    #     yolo = YOLO()
    # else:
    #     yolo = YOLO_ONNX()
    
    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image,c = yolo.detect_image(image, crop = crop, count=count)
            
                # draw=ImageDraw.Draw(image)
                # draw.
                name=0
                while(os.path.exists(os.path.join('img_out',str(name)+'.jpg'))):
                    name+=1
                draw = ImageDraw.Draw(r_image)
                font4count= ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(1e-2 * r_image.size[1] + 30).astype('int32'))
                
                draw.text([20,20], "Number of shrimps: " + str(c), fill=(0, 255, 0), font=font4count)
                # frame = np.array(r_image)
                # # RGBtoBGR满足opencv显示格式
                # r_image = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
                # r_image = cv2.putText(r_image, "Number of shrimps: " + str(c), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                r_image.save(os.path.join('img_out',str(name)+'.jpg'))
                r_image.show()

    elif mode == "Grad_CAM_predict":
        print('in')
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
                print('1')
            except:
                print('Open Error! Try again!')
                continue
            else:
                print('2')
                r_image,c = yolo.detect_heatmap(image,'C:\\code\\yolov7_shrimp\\shrimp_heatmap.jpg')
            
                # draw=ImageDraw.Draw(image)
                # draw.
                # name=0
                # while(os.path.exists(os.path.join('img_out',str(name)+'.jpg'))):
                #     name+=1
                # print('in')
                # r_image.save(os.path.join('img_out',str(name)+'.jpg'))
                # r_image.show()

    elif mode == "merge_predict":

        input_directory = "temp"  # 切割后图像的目录
        output_image_path = "img_out"  # 合并后图像的路径
        # if os.path.exists(input_directory) !=1:
        #     os.mkdir(input_directory)
        
        while True:
            counter=0
            img = input('Input image filename:')
            
            # try:
            #     # image = Image.open(img)
            split_image(img, input_directory, m_rows, m_rows)

                
            # except:
            #     print('Open Error! Try again!')
            #     continue

            # else:
            
            for i in os.listdir(input_directory):
                image = Image.open(os.path.join(input_directory,i))
                r_image,count_plt = yolo.detect_image(image, crop = crop, count=count)
                
                r_image.save(os.path.join(input_directory,i), quality=95, subsampling=0)
                # r_image.show()
                counter+=count_plt
            merge_images(input_directory, os.path.join(output_image_path,os.path.basename(img)), m_rows, m_rows)
            print(os.path.join(output_image_path,os.path.basename(img)))
            image_count=Image.open(os.path.join(output_image_path,os.path.basename(img)))
            draw = ImageDraw.Draw(image_count)
            font4count= ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(1e-2 * image_count.size[1] + 30).astype('int32'))
            
            draw.text([20,20], "Number of shrimps: " + str(counter), fill=(0, 255, 0), font=font4count)
            image_count.save(os.path.join(output_image_path,os.path.basename(img)))
            del draw
            for file in os.listdir(input_directory):
                file_path = os.path.join(input_directory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    elif mode == "video":
        all_count=[]
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        fps_count=[]
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            a , b =yolo.detect_image(frame,count=True)
            frame = np.array(a)
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            fps_count.append(fps)
            # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2.putText(frame, "Number of shrimps: " + str(b), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            all_count.append(b)
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        
        
        print("Video Detection Done!")
        
        capture.release()

        print('fps: %.2f'%(sum(fps_count)/len(fps_count)))
        print("all_count:",all_count)
        print("median:",statistics.median(all_count))
        print("mode:",statistics.mode(all_count))
        print("min:",min(all_count))
        print("max:",max(all_count))
        with open(video_save_path+'.txt', 'w') as f:
            f.write('median: %d\n'%(statistics.median(all_count)))
            f.write('mode: %d\n'%(statistics.mode(all_count)))
            f.write('min: %d\n'%(min(all_count)))
            f.write('max: %d\n'%(max(all_count)))
            f.write('fps: %.2f\n'%(sum(fps_count)/len(fps_count)))
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
