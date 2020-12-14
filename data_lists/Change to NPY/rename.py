import os

FJoin = os.path.join

def renameDataTest(pathFolder):
        isSucces ='False'
        file_list = []
        for dir, subdirs, files in os.walk(pathFolder):
            file_list.extend([FJoin(dir, f) for f in files])

        print(file_list)
        for pathFile in file_list:

            array_temp = pathFile.split('\\')

            if len(array_temp)==4:
                destination = array_temp[0] + '\\' + array_temp[1] + '\\' + array_temp[2] + '\\' + \
                              array_temp[1] + '-' + array_temp[2] + '-' + array_temp[3]
                os.rename(pathFile, destination)
            isSucces='True'
        print(isSucces)

if __name__ == '__main__':
    renameDataTest("vox1_test_wav/wav")






