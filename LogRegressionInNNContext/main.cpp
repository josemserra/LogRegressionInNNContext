#include <Windows.h> 
#include <iostream>

#include <vector>
#include <string>

#include "CImg.h"
#include <Eigen/Dense>

using namespace cimg_library;

// Returns a vector with a path for all the image files in the specified folder
std::vector<std::string> FindAllImgInFolder(std::string folder) {

	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind;

	std::vector<std::string> returnVal;
	std::string newPath = folder;
	newPath.append("\\*.jpg");

	hFind = FindFirstFileA(newPath.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		printf("FindFirstFile failed (%d)\n", GetLastError());
		return returnVal;
	}
	else
	{
		do
		{
			//ignore current and parent directories
			if (strcmp(FindFileData.cFileName, ".") == 0 || strcmp(FindFileData.cFileName, "..") == 0)
				continue;

			if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				//ignore directories
			}
			else
			{
				//list the Files
				std::string temp = folder;
				temp.append("/");
				temp.append(FindFileData.cFileName);
				returnVal.push_back(temp);
			}
		} while (FindNextFile(hFind, &FindFileData));
		FindClose(hFind);
	}

	return returnVal;
}

//Converts an image in CImg format to Eigen and returns it flattened (X,1)
Eigen::MatrixXd convertImg2Eigen(CImg<unsigned char> img) {

	int numPixels = img.height() * img.width();

	Eigen::MatrixXd channelR(img.height(), img.width());
	Eigen::MatrixXd channelG(img.height(), img.width());
	Eigen::MatrixXd channelB(img.height(), img.width());
	Eigen::MatrixXd returnValue(numPixels * 3,1);

	//read into eigen mat
	cimg_forXY(img, colIdx, rowIdx) { 
		channelR(rowIdx, colIdx) = img(colIdx, rowIdx, 0, 0); //Red
		channelG(rowIdx, colIdx) = img(colIdx, rowIdx, 0, 1); //Green	
		channelB(rowIdx, colIdx) = img(colIdx, rowIdx, 0, 2); //Blue
	}

	//flatten Channels
	channelR.resize(numPixels, 1);
	channelG.resize(numPixels, 1);
	channelB.resize(numPixels, 1);

	//Assign the blocks
	returnValue.block(0, 0, numPixels, 1) = channelR; //From row 0 to col (img.height() * img.width()=numPixels) is the R channel
	returnValue.block(numPixels, 0, numPixels, 1) = channelG; //From row numPixels to col 2*numPixels is the G channel
	returnValue.block(2 * numPixels, 0, numPixels, 1) = channelB; //From row 2*numPixels to col 3*numPixels is the B channel

	return returnValue;
}

//Full preprocess of an image, that is load, resize and convert to eigen
Eigen::MatrixXd PreProcessImg(std::string imgFilePath, int imgRescaleValue) {

	//Load img cimg
	CImg<unsigned char> image(imgFilePath.c_str());
	//Resize img
	image.resize(imgRescaleValue, imgRescaleValue);
	//Convert to Eigen + flatten
	Eigen::MatrixXd flattenImg = convertImg2Eigen(image);

	return flattenImg;
}

//Loads a training/dev or test set for a simple binary classification task. You can specify the folder for each class and a rescale value to preprocess the images
//It returns the training samples as a <flattenedImgDim,NumSamples> matrix and a <numSamples,1> mat with the classes of the loaded images
void LoadSet(std::string classExamplesFolder, std::string nonClassExamplesFolder, int imgRescaleValue, Eigen::MatrixXd &outTrainingSamples, Eigen::MatrixXi &outTrainingSamplesClasses) {

	std::vector<std::string> trainImgFilesClass = FindAllImgInFolder(classExamplesFolder);
	std::vector<std::string> trainImgFilesNotClass = FindAllImgInFolder(nonClassExamplesFolder);

	outTrainingSamples = Eigen::MatrixXd(imgRescaleValue*imgRescaleValue*3, trainImgFilesClass.size() + trainImgFilesNotClass.size());
	outTrainingSamplesClasses = Eigen::MatrixXi(1, trainImgFilesClass.size() + trainImgFilesNotClass.size());

	for (int imgIdx = 0; imgIdx < trainImgFilesClass.size(); imgIdx++) {
		//Preprocess Image
		Eigen::MatrixXd flattenImg = PreProcessImg(trainImgFilesClass[imgIdx], imgRescaleValue);
		//Add to trainingSamples
		outTrainingSamples.col(imgIdx) = flattenImg;
		//Create labels
		outTrainingSamplesClasses(0, imgIdx) = 1;
	}

	for (int imgIdx = 0; imgIdx < trainImgFilesNotClass.size(); imgIdx++) {
		//Load img cimg
		Eigen::MatrixXd flattenImg = PreProcessImg(trainImgFilesNotClass[imgIdx], imgRescaleValue);
		//Add to trainingSamples
		outTrainingSamples.col(trainImgFilesClass.size() + imgIdx) = flattenImg;
		//Create labels
		outTrainingSamplesClasses(0, trainImgFilesClass.size() + imgIdx) = 0;

	}

}





int main() {

	std::string trainFolderDogs = "../Img/Train/Dogs";
	std::string trainFolderNotDogs = "../Img/Train/Not Dogs";
	std::string devFolderDogs = "../Img/Dev/Dogs";
	std::string devFolderNotDogs = "../Img/Dev/Not Dogs";
	std::string testFolder = "../Img/Test";

	int imgRescaleValue = 64;

	Eigen::MatrixXd TrainingSamples;
	Eigen::MatrixXi TrainingSamplesClasses;
	//Class 1 - Dogs
	//Class 2 - Not Dogs
	LoadSet(trainFolderDogs, trainFolderNotDogs, imgRescaleValue, TrainingSamples, TrainingSamplesClasses);

	Eigen::MatrixXd DevSamples;
	Eigen::MatrixXi DevSamplesClasses;
	//Class 1 - Dogs
	//Class 2 - Not Dogs
	LoadSet(devFolderDogs, devFolderNotDogs, imgRescaleValue, DevSamples, DevSamplesClasses);






	//Eigen Hello World
	//MatrixXd m(2, 2);
	//m(0, 0) = 3;
	//m(1, 0) = 2.5;
	//m(0, 1) = -1;
	//m(1, 1) = m(1, 0) + m(0, 1);
	//std::cout << m << std::endl;

	////CImg Hello World
	//CImg<unsigned char> image("../Img/lena.jpg"), visu(500, 400, 1, 3, 0);
	//const unsigned char red[] = { 255,0,0 }, green[] = { 0,255,0 }, blue[] = { 0,0,255 };
	//CImgDisplay main_disp(image, "Click a point"), draw_disp(visu, "Intensity profile");
	//while (!main_disp.is_closed() && !draw_disp.is_closed()) {
	//	main_disp.wait();
	//	if (main_disp.button() && main_disp.mouse_y() >= 0) {
	//		const int y = main_disp.mouse_y();
	//		visu.fill(0).draw_graph(image.get_crop(0, y, 0, 0, image.width() - 1, y, 0, 0), red, 1, 1, 0, 255, 0);
	//		visu.draw_graph(image.get_crop(0, y, 0, 1, image.width() - 1, y, 0, 1), green, 1, 1, 0, 255, 0);
	//		visu.draw_graph(image.get_crop(0, y, 0, 2, image.width() - 1, y, 0, 2), blue, 1, 1, 0, 255, 0).display(draw_disp);
	//	}
	//}

	return 0;
}