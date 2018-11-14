import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

current_dir = './Images/001'

# Directory where the data will reside, relative to 'darknet.exe'
#path_data = './NFPAdataset/'

# k fold method

k = 5;

# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt	5 times since k = 5
file_trainA = open('./kfoldtxt/trainA.txt', 'w')  
file_testA = open('./kfoldtxt/testA.txt', 'w')

file_trainB = open('./kfoldtxt/trainB.txt', 'w')  
file_testB = open('./kfoldtxt/testB.txt', 'w')

file_trainC = open('./kfoldtxt/trainC.txt', 'w')  
file_testC = open('./kfoldtxt/testC.txt', 'w')

file_trainD = open('./kfoldtxt/trainD.txt', 'w')  
file_testD = open('./kfoldtxt/testD.txt', 'w')

file_trainE = open('./kfoldtxt/trainE.txt', 'w')  
file_testE = open('./kfoldtxt/testE.txt', 'w')

# Populate train.txt and test.txt
counter = 1  
index_test = round(100 / percentage_test)

print("index test is:", index_test)

# Allocate the images to 5 different sets, each with 106 samples since we have total of 530 images   
#for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
#    title, ext = os.path.splitext(os.path.basename(pathAndFilename))	

#    if counter == index_test:
#        counter = 1
#        file_testA.write(current_dir + "/" + title + '.jpg' + "\n")
#    else:
#        file_trainA.write(current_dir + "/" + title + '.jpg' + "\n")
#        counter = counter + 1

index = 0

# write for A
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    filesp = title.split("_");
    # print("The number of the file is: ", filesp[3])
    num = int(filesp[3])
    # print("The real int of the file is: " , num)
    
    # A
    if(num > 1 and num < 107):
        file_testA.write(current_dir + "/" + title + '.jpg' + "\n")
    else:
        file_trainA.write(current_dir + "/" + title + '.jpg' + "\n")
    # B
    if(num >= 107 and num < 213):
        file_testB.write(current_dir + "/" + title + '.jpg' + "\n")
    else:
        file_trainB.write(current_dir + "/" + title + '.jpg' + "\n")
    
    # C
    if(num >=213 and num < 319):
        file_testC.write(current_dir + "/" + title + '.jpg' + "\n")
    else:
        file_trainC.write(current_dir + "/" + title + '.jpg' + "\n")
    
    # D
    if(num >=319 and num < 425):
        file_testD.write(current_dir + "/" + title + '.jpg' + "\n")
    else:
        file_trainD.write(current_dir + "/" + title + '.jpg' + "\n")
    
    # E
    if(num >= 425 and num <= 530):
        file_testE.write(current_dir + "/" + title + '.jpg' + "\n")
    else:
        file_trainE.write(current_dir + "/" + title + '.jpg' + "\n")
        
    