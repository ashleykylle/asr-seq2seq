FOR TRAINING THE MODEL:

Run pathmaker.py make sure to change the corresponding directories for; 
    1 - the directory where all your .wav files are make sure the structure of the folder is as follows
        ROOT
            -speaker
                -rec_subfolder
                    -wav files
            -speaker
                -rec_subfolder
                    -wav files
    2 - the path to the csv file where the transcriptions for the wav files are make sure the format it "filename,transcriptions" the ordering also has to match the ordering of the wav files in the folder so wav file on the first row in your file explorer will match
    first line in the csv

After successfully running pathMaker.py it should generate a file called transcription_updated.csv

Open the finetunedModel.ipynb and change the corresponding paths matching in your local directory;
    1 - the transcription_updated.csv
    2 - the corresponding wav file directory for evaluating the model make sure the structure of the folder is as follows
        ROOT
            - wav files
    3 - the corresponding transcriptions for the wav files to be evaluated

Uncomment the commented out codes in the notebook then run all

NOTE: Training will take about 4-10 hours depending on your cpu and gpu
