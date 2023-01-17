from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np
import os 
import tensorflow as tf
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


class AI_model():
    
    def __init__(self,model,model2,model3,model4,model5,model6,model7,model8):
        # Load the model
        self.model = load_model(model) #compile=False)
        self.model2 = load_model(model2) #compile=False)
        self.model3 = load_model(model3) #compile=False)
        self.model4 = load_model(model4) #compile=False)
        self.model5 = load_model(model5) #compile=False)
        self.model6 = load_model(model6) #compile=False)
        self.model7 = load_model(model7) #compile=False)
        self.model8 = load_model(model8) #compile=False)
        # Load the labels

        
    def predict(self,img_file):
    
        '''image=self.get_input_image(img_file)
        image2=self.get_input_image2(img_file)
        image3=self.get_input_image3(img_file)
        image4=self.get_input_image4(img_file)
        image5=self.get_input_image5(img_file)'''
        confidence_scores=[]
        
        for i in range(4):
            if i == 0:
                size,shape,preprocessing_funct,self.model=(255,255),(1, 255, 255, 3),tf.keras.applications.efficientnet_v2.preprocess_input,self.model
            elif i==1:
                size,shape,preprocessing_funct,self.model=(128,128),(1, 128, 128, 3),None,self.model2
            elif i==2:
                size,shape,preprocessing_funct,self.model=(128,128),(1, 128, 128, 3),tf.keras.applications.inception_v3.preprocess_input,self.model3
            elif i==3:
                size,shape,preprocessing_funct,self.model=(224,224),(1, 224, 224, 3),tf.keras.applications.resnet_v2.preprocess_input,self.model4
            image = Image.open(img_file).convert('RGB')

            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            

            data = np.ndarray(shape, dtype=np.float32)

            #turn the image into a numpy array
            image_array = np.asarray(image)
            image_array = preprocessing_funct(image_array) if preprocessing_funct  else image_array
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 255.0)

            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            prediction = self.model.predict(data)
		
            index = np.argmax(prediction)

            print(prediction)
            confidence_score = prediction[0][index]#self.class_names
            confidence_scores.append(confidence_score)
        probx = sum(confidence_scores)/4		
        if probx<.3:
            label="NORMAL"
        else:
            confidence_scorex=[]

            for j in range(4):
                if j == 0:
                    size,shape,preprocessing_funct,self.model=(256,256),(1, 256, 256, 3),None,self.model5
                elif j==1:
                    size,shape,preprocessing_funct,self.model=(224,224),(1, 224, 224, 3),tf.keras.applications.resnet_v2.preprocess_input,self.model6
                elif j==2:
                    size,shape,preprocessing_funct,self.model=(224,224),(1, 224, 224, 3),tf.keras.applications.mobilenet.preprocess_input,self.model7
                elif j==3:
                    size,shape,preprocessing_funct,self.model=(224,224),(1, 224, 224, 3),tf.keras.applications.resnet_v2.preprocess_input ,self.model8 
                image = Image.open(img_file).convert('RGB')

                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                

                data = np.ndarray(shape, dtype=np.float32)

                #turn the image into a numpy array
                image_array = np.asarray(image)
                image_array = preprocessing_funct(image_array) if preprocessing_funct  else image_array
                # Normalize the image
                normalized_image_array = (image_array.astype(np.float32) / 255.0)

                # Load the image into the array
                data[0] = normalized_image_array

                # run the inference
                prediction = self.model.predict(data)
		
                index = np.argmax(prediction)

                print(prediction)
                confidence_score = prediction[0][index]#self.class_names
                confidence_scorex.append(confidence_score)
            proby = sum(confidence_scorex)/4		
            if proby>.5:
                label = "VIRUS"
            else:
                label="BACTERIA"
            #confidence_scorex=(confidence_score+confidence_score2+confidence_score3+confidence_score4)/4
        
          
			
        return probx,label
    
    


if __name__ == '__main__':  

    model_dir='model'
    model_file='ChestX-Ray2_eff_acc95_rec97_prec95.h5'
    model_file2='ChestX-Ray2_cnn_acc93_rec97_prec92.h5'
    model_file3='ChestX-Ray2_incv_acc95_rec97_prec94.h5'
    model_file4='ChestX-Ray2_resnet_acc94_rec96_prec95.h5'
    model_file5='ChestX-Ray2BV_cnn_acc92_rec89_prec89.h5'
    model_file6='ChestX-Ray2BV_ResNet152V2_acc91_rec83_prec93.h5'
    model_file7='ChestX-Ray2BV_mobilenet_acc88_rec77_prec91.h5'
    model_file8='ChestX-Ray2BV_ResNet50V2_acc90_rec85_prec88.h5'
    label_file='labels.txt'
    upload_img='下載.png'
    
  
    model_file=os.path.join(model_dir,model_file)
    model_file2=os.path.join(model_dir,model_file2)
    model_file3=os.path.join(model_dir,model_file3)
    model_file4=os.path.join(model_dir,model_file4)
    class_file=os.path.join(model_dir,label_file)
    model=AI_model(model_file,model_file2,model_file3,model_file4,class_file)

    
    img_file=os.path.join('upload',upload_img)
    confx,label,conf,conf2,conf3,conf4=model.predict(img_file)
    
    print('Class:', label, end='')
    print('Confidence score:', confx,",",conf,",",conf2,",",conf3,",",conf4)

