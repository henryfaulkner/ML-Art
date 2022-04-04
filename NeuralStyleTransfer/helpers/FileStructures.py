import os
import io
import matplotlib.pyplot as plt
import requests
from MetAPI.RESTCalls import RESTCalls


class FileStructures:
    def DownloadImageFromMET(self, metJson, folderName):
        """ Method

        """

        result = {
            'Title': metJson['title'],
            'Artist': metJson['artistDisplayName'],
            'Image': metJson['primaryImage']
        }

        url = metJson['primaryImage']
        imageFile = requests.get(url)
        fileExtension = url.split('/')[-1].split('.')[-1]
        fileName = result['Title'] + '.' + fileExtension
        filePath = os.path.join(
            os.getcwd(), "../Images/", folderName, fileName)

        with open(filePath, 'wb') as file:
            file.write(imageFile.content)

        result['FileName'] = fileName

        return result

    def CreateResultFolder(self, folder_and_file_name):
        try:
            # Create target Directory
            cwd = os.getcwd()
            dir = os.path.join(cwd, "../Images/", folder_and_file_name)
            os.makedirs(dir)
            print("Directory ", folder_and_file_name,  " Created ")
        except FileExistsError:
            print("Directory ", folder_and_file_name,  " already exists")

        return

    def RenameVideoIfManual(self, folder_and_file_name, content_style_tuple):
        if(folder_and_file_name != "temp"):
            return folder_and_file_name

        content = content_style_tuple[2]['Title'].replace(' ', '_')
        style = content_style_tuple[3]['Title'].replace(' ', '_')
        return content + "+" + style

    def RenameResultFolderName(self, folder_and_file_name, content_style_tuple):
        curDir = os.path.join(os.getcwd(), "../Images/", folder_and_file_name)

        content = content_style_tuple[2]['Title'].replace(' ', '_')
        style = content_style_tuple[3]['Title'].replace(' ', '_')
        result = content + "+" + style

        newDir = os.path.join(os.getcwd(), "../Images/", result)

        os.rename(curDir, newDir)

        return result

    def SaveResultFile(self, best_img, folder_and_file_name, numIterations):
        cwd = os.getcwd()
        filename = '(' + str(numIterations) + ')' + '.png'
        filepath = os.path.join(
            cwd, "../Images/", folder_and_file_name, filename)

        with open(filepath, 'wb'):
            fig = plt.figure(figsize=(10, 10), frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(best_img, aspect='auto')
            plt.savefig(filepath)

        return

    def CreateFileInput(self):
        """" This method preforms UI for Output file creation """
        yes_or_no = input(
            "Do you want to save the end result? (yes(y) or no(n)\n")

        if yes_or_no == "no" or yes_or_no == "n":
            return

        folder_and_file_name = input("Enter your desired structure name: \n")

        return folder_and_file_name

    def AccessImageInput(self, folder_and_file_name):
        """" This method preforms UI for getting an image file """
        yes_or_no = input(
            "Do you want to use MET Gallery images? (yes(y) or no(n)\n")

        if yes_or_no == "no" or yes_or_no == "n":
            content_file_name = input(
                'Enter your content image file\'s name: ')
            style_file_name = input('Enter your style image file\'s name: ')
            content_path = '../Images/Manual_Images/' + content_file_name
            style_path = '../Images/Manual_Images/' + style_file_name
            return (content_path, style_path)

        api = RESTCalls()

        # Content Image
        metJson = api.GetRandomArtDocument()
        content_dict = self.DownloadImageFromMET(metJson, folder_and_file_name)
        content_path = "../Images/" + \
            folder_and_file_name + "/" + content_dict['FileName']

        # Style Image
        metJson = api.GetRandomArtDocument()
        style_dict = self.DownloadImageFromMET(metJson, folder_and_file_name)
        style_path = "../Images/" + \
            folder_and_file_name + "/" + style_dict['FileName']

        return (content_path, style_path, content_dict, style_dict)
