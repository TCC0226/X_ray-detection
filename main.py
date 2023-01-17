# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31, 2022
@author: joseph@艾鍗學院

 www.ittraining.com.tw

"""

from flask import *  
from aimodel import AI_model #joseph
import os 
#app = Flask(__name__)  
app = Flask(__name__, template_folder='templates', static_folder='upload')
# # Define allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save('upload/'+f.filename)  
        
        #predict the image 
        img_file=os.path.join('upload',f.filename)
        probx,label=model.predict(img_file)
        probx=round(100*probx,2)
        print('Class:',label, end='')
		
        print('Confidence score:', probx)
		
        return render_template("success.html", name = img_file,class_name=label,probability=probx)  


            

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


    model_file=os.path.join(model_dir,model_file)
    model_file2=os.path.join(model_dir,model_file2)
    model_file3=os.path.join(model_dir,model_file3)
    model_file4=os.path.join(model_dir,model_file4)
    model_file5=os.path.join(model_dir,model_file5)
    model_file6=os.path.join(model_dir,model_file6)
    model_file7=os.path.join(model_dir,model_file7)
    model_file8=os.path.join(model_dir,model_file8)

    model=AI_model(model_file,model_file2,model_file3,model_file4,model_file5,model_file6,model_file7,model_file8)

    

    #run flask web engine
    app.run(host= '0.0.0.0', port=3100 ,debug = True) 
 

    
     
