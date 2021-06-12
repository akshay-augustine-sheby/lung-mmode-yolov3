import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import base64
import streamlit.components.v1 as components

def prediction(x):
  if x[0]==1 and x[5]==1:
    return("Consolidation")
  elif x[1]==1 and x[5]==1:
    return("Pleural Effusion")
  elif x[2]==1 and x[3]==1 and x[4]==1:
    return("Normal Lung")
  else:
    return("Error")

def detect_objects(our_image):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.beta_columns(2)

    col1.subheader("Original Image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(our_image)
    col1.pyplot(use_column_width=True)

    # YOLO ALGORITHM
    import urllib.request

    url1 = 'https://github.com/akshay-augustine-sheby/mmode-yolov3/releases/tag/others/yolov3_training_last.weights'
    hf1 = url1.split('/')[-1]

    urllib.request.urlretrieve(url1, hf1)
    
    url2 = 'https://github.com/akshay-augustine-sheby/mmode-yolov3/releases/tag/others/yolov3_testing.cfg'
    hf2 = url2.split('/')[-1]

    urllib.request.urlretrieve(url2, hf2)
    
    
    
    net = cv2.dnn.readNet(hf1, hf2)

    classes = ["irr p line","sine p line","cont p line","seashore","a line","fluid"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(len(classes), 3))


    # LOAD THE IMAGE
    our_image = np.array(our_image.convert('RGB'))
    img = cv2.resize(our_image, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape


    # DETECTING OBJECTS (CONVERTING INTO BLOB)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   #(image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outs = net.forward(output_layers)


    arr=[0,0,0,0,0,0]    #["irr p line","sine p line","cont p line","seashore","a line","fluid"]


    class_ids = []
    confidences = []
    boxes =[]

    # SHOWING INFORMATION CONTAINED IN 'outs' VARIABLE ON THE SCREEN
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores,axis=0)
            confidence = scores[class_id]
            #st.write(class_id)
            if confidence > 0.3:
                # OBJECT DETECTED
                arr[class_id]=1
                #Get the coordinates of object: center,width,height
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  #width is the original width of image
                h = int(detection[3] * height) #height is the original height of the image

                # RECTANGLE COORDINATES
                x = int(center_x - w /2)   #Top-Left x
                y = int(center_y - h/2)   #Top-left y

                #To organize the objects in array so that we can extract them later
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.4)
    #print(indexes)

    font = cv2.FONT_HERSHEY_PLAIN
    items=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            if(x>10):
              cv2.putText(img, label, (x, y + 30), font, 2, color, 2)
            else:
              cv2.putText(img, label, (x+20, y + 30), font, 2, color, 2)
            items.append(label)

    st.text("")
    col2.subheader("Object-Detected Image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    col2.pyplot(use_column_width=True)

    st.success("The scan is of {}".format(prediction(arr)))


def object_main():
    """YOLO V3"""

    st.title("M-mode YOLOv3")
    st.write("YOLOv3 (You Only Look Once, Version 3) is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images.")

    #choice = st.radio("", ("Show Demo", "Browse an Image"))
    #st.write()

    st.set_option('deprecation.showfileUploaderEncoding', False)
    image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)
        detect_objects(our_image)

st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)

page_bg_img = '''
<style>
body {
background-image: "stream.jpg";
background-size: cover;

}
</style>


'''
main_bg = "stream.jpg"
main_bg_ext = "jpg"

side_bg = "stream.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
  
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True)



st.markdown('<h1 style="float: full;background-color:black">M-mode Lung Ultrasound Pattern Recognition</h1><img style="float: right;" src=".jpg" />', unsafe_allow_html=True)


if __name__ == '__main__':
    object_main()
