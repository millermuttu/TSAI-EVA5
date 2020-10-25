#NEW FEATURES
#-------------------------------------------------------------------------------
# Name:        Outputting format for orders
# Purpose:     Outputting format for orders training yoloV3
# Author:      aka9
# Created:     28/07/2019

#
#-------------------------------------------------------------------------------

import glob, os, sys

current_dir = '/content/YoloV3/data/YoloV3_Dataset/images'

def process(current_dir):
    #current_dir = 'Images/full_catAndDog'

    # Percentage of images to be used for the test set
    percentage_test = 10;

    # Create and/or truncate train.txt and test.txt
    file_train = open('train.txt', 'w')
    file_test = open('test.txt', 'w')

    # Populate train.txt and test.txt
    counter = 1
    index_test = round(100 / percentage_test)
    print("Warning: we assume that all pictures to be processed are jpgs")
    for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test:
            counter = 1
            file_test.write(current_dir + "/" + title + '.jpg' + "\n")
        else:
            file_train.write(current_dir + "/" + title + '.jpg' + "\n")
            counter = counter + 1

def main():
  process(current_dir)

if __name__ == '__main__':
    main()
