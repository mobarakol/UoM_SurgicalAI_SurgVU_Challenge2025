# Dataset Building 

### Step 1:
Run train_dataset.py to create the dataset by sampling randomly to obtain 2 clips for each case. Each clips is 30 seconds long. Clips that did not have the DaVinci interface or any sort of unnatural occlusions were exlcluded after manual checking. 

### Step 2:
Run train_stat.py to see the frequency of each unique category. Given that the clips created in Step 1 may have caused no selection of any unique case or after manual checking it may of had to be removed due to occlusions or no DaVinci interface. You can use clip_maker.py to manually extract 30 second time sequences to include at least one instance of an unique category after which manually create a csv file as created by Step 1. Also, some files would be mvoed from train to validation to ensure each unique category appears at least once in each, to identify which cases have which unique category allowing moving of correct files. 

### Step 3:
Sample QAs were created to make our own dataset:

Is a $tool_name$ among the listed tools? | No, a $tool_name$ is not listed. / Yes, a $tool_name$ is not listed.

Was a $tool_name$ used in this clip? | No, a $tool_name$ was not utilized. / Yes, a $tool_name$ was not utilized.

Was a $tool_name$ used during the surgery? | No, a $tool_name$ was not used. / Yes, a $tool_name$ was not used.

Are there forceps being used here? | No, forceps are not mentioned. / Yes, forceps are mentioned.

What type of forceps are used? |  / Bipolar forceps

What task is being performed in this clip? | The summary is describing endoscopic or laparoscopic surgery.

Run train_labelling.py to label each clip following this template. After manually creating the csv file for validation set that was provided you can run val_labelling.py to achieve the template QAs for them. 
