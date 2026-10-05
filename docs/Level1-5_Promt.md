
################################################################################
################################################################################ 
## '''Level 1'''

## Level1-1 : "task": "noise_recognition"

    "type": "wrong_question"  ## 反例
    "type": "correct_question"  ## 正例
    "type": "normal_question"  ##


[//]: # (example)
"image_paths": "xxx/path", 
"question": "Which of these two images shows the added noise?", 
"options": ["A. left", "B. right"],
"answer": "A. left",


## Level1-2 : "task": "artifact_recognition"

    "type": "wrong_question"  ## 反例
    "type": "correct_question"  ## 正例
    "type": "normal_question"  ##

"image_paths": "xxx/path", 
"question": "Here are two medical images. One of them has added artifact. Is the left image added artifact?", 
"options": ["A. yes", "B. no"], 
"answer": "A. yes",



################################################################################
################################################################################ 
## '''Level 3'''

## Level3-1 : "task": "organ_location"

    "type": "wrong_color-correct_region"  ## 反例
    "type": "correct_color-correct_region"  ## 正例
    type+1:  "question_type": "normal_question" 四选一

有三个部分 其中一个部分是 open-question 其余两个是AB 单选 2选1
[//]: # (example - close question)
"image_paths": "xxx/path", 
"question": "Is the stomach represented by Limegreen in this image?", 
"options": ["A. yes", "B. no"], 
"answer": "B. no"

[//]: # (example - open question)
"image_paths": "xxx/path", 
"question": "What color represents right kidney in this image?", 
"options": ["A. Green", "B. Blue", "C. Red", "D. Magenta"], 
"answer": "B. Blue",
"question_type": "normal_question"}




## Level3-2 : "task": "organ_recognition"

    "type": "wrong_box-correct_Q"  ## 反例
    "type": "correct_box-correct_Q"  ## 正例
    "type": "correct_box_wrong_Q", ## 反例

[//]: # (example)
"image_paths": "xxx/path"
"question": "Is the mediastinum present within the red bounding box in this scan?", 
"options": ["A. Yes ", "B. No"], 
"answer": "B. No",



## Level3-3 : "task": "organ_eraser_identification"
    "modality": "Radiograph",
    "modality": "CT"
    "modality": "Magnetic_Resonance_Imaging"
四个模态 = 同一个task 不用分类

四个选项：
[//]: # (example)
"image_paths": "xxx/path"
"question": "In the given pair of CT images, the original scan is on the left and the AI-modified version is on the right. Which organ is missing or altered in the right image?", 
"options": ["A. left lung", "B. right lung", "C. heart", "D. first rib"], 
"answer": "D. first rib",




################################################################################
################################################################################ 
## '''Level 4'''

## Level4-1 : "task": "bodypart_recognition"

    "type": "wrong_box-correct_Q"  ## 反例
    "type": "correct_box-correct_Q"  ## 正例
    "type": "correct_box-wrong_Q"  ## 正例

[//]: # (example)
"image_paths": "xxx/path",
"question": "Is there a lesion in the lung within this red box?", 
"options": ["A. Yes, there is a lesion in the red box.", "B. No, there is no lesions in the red box."], 
"answer": "A. Yes, there is a lesion in the red box.",



## Level4-2 : "task": "lesion_recognition"

     "type": "correct_box-correct_Q", ## 正例
     "type": "wrong_box-correct_Q"  ## 反例
     "type": "correct_box-wrong_Q"  ## 反例

[//]: # (example)
"image_paths": "xxx/path"
"question": "Based on this scan, does the red boxed finding suggest any peripheral ground glass densities?", 
"options": ["A. yes", "B. no"], 
"answer": "A. yes",

## Level4-3 : "task": "lesion_multi_box"
     属性：   "question_type": "3options",

# 改成选择题 三选一 红黄蓝

[//]: # (example)
"question": "Among the colored boxes, which one encloses the lymph nodes measuring 7 mm in the short axis of the largest in the mediastinal upper-lower paratracheal, prevascular area in the aortopulmonary window and in the subcarinal localization?", 
"opotions" : A.red B.blue C.yellow
"answer": "A.red",
"image_paths": "xxx/path"


## Level4-4 : "task": "lesion_size"
    "type": "wrong_size-wrong_Q",  ## 反例
    "type": "correct_size-correct_Q"  ## 正例
    
    属性：
    "question_type": "open_question"
[//]: # (example)
"image_paths": "xxx/path"
"question": "Is the parenchymal air cyst highlighted in red box at the right lung upper lobe approximate measured at 2 cm? If not, what is the correct measurement?", 
"answer": "No, the parenchymal air cyst at the right lung upper lobe measures approximately 1 cm, not 2 cm. ",




################################################################################
################################################################################
## '''Level 5  : Report Generation'''

## Level5-1 : "task": "box_report"

    "type": "add_box"  ## 正例
    "type": "no_box"  ## 反例

    "question_type": "open_question"

[//]: # (example)
"image_paths": "xxx/path"
"question": "Generate a clinical report for this CT scan. ", 
"answer": "When examined in the lung parenchyma window; no mass nodule-infiltration was detected in both lung parenchyma.",



## Level5-2 : "task": "noise_report"

    "type": "correct_box_no_noise"  ## 正例
    "type": "correct_box_add_noise"  ## 反例

    "question_type": "open_question"

[//]: # (example)
"image_paths": "xxx/path"
"question": "Generate a clinical report for this CT scan.", 
"answer": "No pathological increase in wall thickness is observed in the thoracic esophagus."


## Level5-3 : "task": "artifact_report"

    "type": "correct_box_no_artifact"  ## 正例
    "type": "correct_box_add_artifact"  ## 反例

    "question_type": "open_question"

[//]: # (example)
"image_paths": "xxx/path"
"question": "Generate a clinical report for this CT scan.", 
"answer": "Consolidative density increases are observed in the lower lobe of both lungs and are accompanied by peribronchial thickening. Nonspecific nodules less than 3 mm were observed in both lungs. There are pleuroparenchymal fibrotic sequelae bands in the right lung middle lobe medial and left lung lingular segment."


################################################################################
################################################################################ 
## '''Joint QA '''

## Joint question : "task": "joint_qa"

    question1 ：What imaging modality is used in this image?
                该图像采取哪种成像方式？ CT/MRI 二选一
    question2 ：Which organ appears to be abnormal in this image?  
                红框里是什么器官有病变？ 心脏   四选一
    question3 ：Based on the abnormal organ, what lesion or finding is most clearly visible?  
                基于这个不正常的器官，最清晰可见的病变或体征是什么？ 血管瘤  四选一
    question4 ：Considering all the above findings, what is the most likely diagnosis?
                综合以上所有发现，最可能的诊断是什么？ 心脏血管瘤   四选一

    all answer 选择题 
        "modality_options" : options1,
        "modality_answer" : answer1,

        "organ_options" : options2,
        "organ_answer" : answer2,

        "lesion_options" : options3,
        "lesion_answer" : answer3,

        "diagnosis_options" : options4,
        "diagnosis_answer" : answer4,

[//]: # (example)





