#include <Windows.h> 
#include <iostream>
#include <fstream> 

#include <vector>
#include <string>
#include <algorithm> 

#include <time.h>

#include "CImg.h"
#include <Eigen/Dense>

using namespace cimg_library;

void drawPlot(CImgDisplay& disp, std::vector<double> x, std::vector<double> y,
	double minX, double maxX, double minY, double maxY,
	std::string xLabel, std::string yLabel);

//////////////////////////////////////////////////////
// - Step 1 -
// Load files and preprocess the data
//////////////////////////////////////////////////////

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
	Eigen::MatrixXd returnValue(numPixels * 3, 1);

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
	//Normalise. This could be the average image of the full DB, but it works just fine with 255
	flattenImg *= 1.0 / 255;
	return flattenImg;
}

//Loads a training/dev or test set for a simple binary classification task. You can specify the folder for each class and a rescale value to preprocess the images
//It returns the training samples as a <flattenedImgDim,NumSamples> matrix and a <numSamples,1> mat with the classes of the loaded images
void LoadSet(std::string classExamplesFolder, std::string nonClassExamplesFolder, int imgRescaleValue, Eigen::MatrixXd &outTrainingSamples, Eigen::MatrixXi &outTrainingSamplesClasses) {

	std::vector<std::string> trainImgFilesClass = FindAllImgInFolder(classExamplesFolder);
	std::vector<std::string> trainImgFilesNotClass = FindAllImgInFolder(nonClassExamplesFolder);

	outTrainingSamples = Eigen::MatrixXd(imgRescaleValue*imgRescaleValue * 3, trainImgFilesClass.size() + trainImgFilesNotClass.size());
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

//////////////////////////////////////////////////////
// - Step 2 -
// Neuron Initialization, Forward Prop, Sigmoid Activation, Loss and Cost Functions
//////////////////////////////////////////////////////

//Initialize all the weights with random values (small) and b with 0
void InitializeNeuron(int inputSize, Eigen::MatrixXd &weights, Eigen::VectorXd &b) {

	srand(1); // Just to force random to always generate the same randoms (good for tests purposes)

	weights = Eigen::MatrixXd::Random(inputSize*inputSize * 3, 1)*0.01; //keep values small
	b = Eigen::VectorXd::Zero(1);
}

//Activation function. Applies the sigmoid function element wise on a matrix. Changes the input
void Sigmoid(Eigen::MatrixXd &z) {
	z = (1.0 + (-1 * z.array()).exp()).inverse().matrix();
}

//Forward Propagation step for single neuron
Eigen::MatrixXd ForwardPropagation(Eigen::MatrixXd weights, Eigen::VectorXd b, Eigen::MatrixXd X) {

	Eigen::MatrixXd A = weights.transpose()*X;
	A.colwise() += b;

	Sigmoid(A);

	return A;
}

//Cross Entropy Loss Function. A are the predictions, Y are the training labels 
Eigen::MatrixXd CrossEntropy(Eigen::MatrixXd &A, Eigen::MatrixXd &Y) {
	Eigen::MatrixXd entropy = -Y.array()*((A.array()).log()) - (1 - Y.array())*((1 - A.array()).log());
	return entropy;
}

//Calculates the cost for all the samples in A
double CalculateCost(Eigen::MatrixXd A, Eigen::MatrixXd Y) {
	double E = 0.00000001;

	int m = A.cols();
	Eigen::MatrixXd entropy = CrossEntropy(A, Y);
	double cost = (1.0 / m)*(entropy.sum() + E);
	return cost;
}

//////////////////////////////////////////////////////
// - Step 3 -
// Backward Propagation
//////////////////////////////////////////////////////

//Calculates dJ/dW (dw) and dJ/db (db), which describe how much the weights should change to approximate the predictions of the true classes
void BackwardPropagation(Eigen::MatrixXd X, Eigen::MatrixXd A, Eigen::MatrixXd Y, Eigen::MatrixXd &dw, Eigen::MatrixXd &db) {

	Eigen::MatrixXd dz = A - Y;
	int m = dz.cols();

	Eigen::VectorXd dzV(Eigen::Map<Eigen::VectorXd>(dz.data(), m)); //otherwise I can't broadcast in the line below

	dw = X.array().rowwise() * dzV.transpose().array();
	Eigen::MatrixXd dTemp = (1.0 / m)*(dw.rowwise().sum()); // Eigen behaves strangly if I don't store the results in a temp variable
	dw = dTemp;

	dTemp = (1.0 / m)*(dz.rowwise().sum());
	db = dTemp;
}

//////////////////////////////////////////////////////
// - Step 4 -
// (Batch) Gradient Descent
// - Step 5 -
// Added the code on the (Batch) Gradient Descent methods to update the plot
//////////////////////////////////////////////////////

void ShuffleMatrixCols(Eigen::MatrixXd X, Eigen::MatrixXi X_Classes, Eigen::MatrixXd &X_perm, Eigen::MatrixXi &X_Classes_Perm) {

	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X.cols());
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
	X_perm = X * perm; // permute columns
	X_Classes_Perm = X_Classes * perm; // permute columns
									   //Eigen::MatrixXi x2_perm2 = perm * x123; // permute rows

}

void GradientDescent(Eigen::MatrixXd X, Eigen::MatrixXi X_Classes, Eigen::MatrixXd &weights, Eigen::VectorXd &b, int numEpochs = 25, double learningRate = 0.001, bool plotCost = false) {

	CImgDisplay main_disp;
	std::vector<double> x;
	std::vector<double> y;
	if (plotCost)
		main_disp = CImgDisplay(500, 400, "Cost Plot"); // display it


														//Gradient descent
	for (int itIdx = 0; itIdx < numEpochs; itIdx++) {

		Eigen::MatrixXd preds = ForwardPropagation(weights, b, X);

		double cost = CalculateCost(preds, X_Classes.cast <double>());

		if (plotCost) {
			x.push_back(itIdx);
			y.push_back(cost);
			drawPlot(main_disp, x, y,
				0.0f, 15.0f, 0.0f, 1.0f,
				"Iterations", "Cost");
		}

		//Single Back Prop Step
		Eigen::MatrixXd dw;
		Eigen::MatrixXd db;
		BackwardPropagation(X, preds, X_Classes.cast <double>(), dw, db);

		//Update weights
		weights = weights - learningRate*dw;
		b = b - learningRate*db;
	}

	main_disp.wait(); // Wait for key any key input
}

void BatchGradientDescent(Eigen::MatrixXd X, Eigen::MatrixXi X_Classes, Eigen::MatrixXd &weights, Eigen::VectorXd &b, int batchSize = 32, int numEpochs = 25, double learningRate = 0.001, bool plotCost = false) {

	CImgDisplay main_disp;
	std::vector<double> x;
	std::vector<double> y;
	if (plotCost)
		main_disp = CImgDisplay(500, 400, "Cost Plot"); // display it


	int numTrainingSamples = X.cols();
	int counterPlot = 0;
	for (int itIdx = 0; itIdx < numEpochs; itIdx++) {

		//Random shuffle of samples
		Eigen::MatrixXd X_perm;
		Eigen::MatrixXi X_Classes_Perm;
		ShuffleMatrixCols(X, X_Classes, X_perm, X_Classes_Perm);

		//Process all batches aside from last one, which might have a different size than the others
		int processedBatches = 0;
		while ((processedBatches + batchSize) < numTrainingSamples) {

			Eigen::MatrixXd batch = X.block(0, processedBatches, X.rows(), batchSize);
			Eigen::MatrixXi batchClasses = X_Classes.block(0, processedBatches, 1, batchSize);

			Eigen::MatrixXd preds = ForwardPropagation(weights, b, batch);

			double cost = CalculateCost(preds, batchClasses.cast <double>());

			if (plotCost) {
				x.push_back(counterPlot);
				y.push_back(cost);
				drawPlot(main_disp, x, y,
					0.0f, 15.0f, 0.0f, 1.0f,
					"Iterations", "Cost");
				counterPlot++;
			}


			//Single Back Prop Step
			Eigen::MatrixXd dw;
			Eigen::MatrixXd db;
			BackwardPropagation(batch, preds, batchClasses.cast <double>(), dw, db);

			weights = weights - learningRate*dw;
			b = b - learningRate*db;

			processedBatches += batchSize;
		}

		//Process the last batch
		Eigen::MatrixXd batch = X.block(0, processedBatches, X.rows(), numTrainingSamples - processedBatches);
		Eigen::MatrixXi batchClasses = X_Classes.block(0, processedBatches, 1, numTrainingSamples - processedBatches);

		Eigen::MatrixXd preds = ForwardPropagation(weights, b, batch);

		double cost = CalculateCost(preds, batchClasses.cast <double>());

		if (plotCost) {
			x.push_back(counterPlot);
			y.push_back(cost);
			drawPlot(main_disp, x, y,
				0.0f, 15.0f, 0.0f, 1.0f,
				"Iterations", "Cost");
			counterPlot++;
		}


		Eigen::MatrixXd dw;
		Eigen::MatrixXd db;
		BackwardPropagation(batch, preds, batchClasses.cast <double>(), dw, db);

		//Update weights
		weights = weights - learningRate*dw;
		b = b - learningRate*db;

	}

	main_disp.wait(); // Wait for key any key input
}

//////////////////////////////////////////////////////
// - Step 5 -
// Visualise Cost after each Gradient Descent Step
//////////////////////////////////////////////////////
void drawPlot(CImgDisplay& disp, std::vector<double> x, std::vector<double> y,
	double minX = 0.0f, double maxX = 15.0f, double minY = 0.0f, double maxY = 1.0f,
	std::string xLabel = "xAxis", std::string yLabel = "yAxis") {

	const unsigned char lineColour[] = { 0,0,0 };// i.e. black
	int bgFillColour = 255;// i.e. white

	if (x.size() != y.size()) {
		std::cout << "Both vectors need to have the same size. \n Will not draw anything \n";
	}

	int dispWidth = disp.width();
	int dispHeight = disp.height();

	CImg<unsigned char>  visu(dispWidth, dispHeight, 1, 3, 1);

	//Plot Drawing limits, i.e. what is drawn inside the axis lines
	int xMinAxis = 50;
	int yMinAxis = 10;
	int xMaxAxis = dispWidth - 10;
	int yMaxAxis = dispHeight - 50;

	//Validate the max. if any of the values in x or y are larger or smaller than the the max or min (respectively), increase the max and decrease the min (respectively).
	auto minX_it = std::min_element(std::begin(x), std::end(x));
	auto maxX_it = std::max_element(std::begin(x), std::end(x));
	auto minY_it = std::min_element(std::begin(y), std::end(y));
	auto maxY_it = std::max_element(std::begin(y), std::end(y));

	if (*minX_it < minX)
		minX = *minX_it - 1;
	if (*maxX_it > maxX) {
		maxX = *maxX_it + 1;
	}
	if (*minY_it < minY)
		minY = *minY_it - 1;
	if (*maxY_it > maxY)
		maxY = *maxY_it + 1;


	visu.fill(bgFillColour);

	//Draw Axis labels
	visu.rotate(90);
	visu.draw_text((int)dispHeight / 2, 10, yLabel.c_str(), lineColour, 0, 1, 30, 30);
	visu.rotate(-90);
	visu.draw_text((int)(dispWidth / 2 - (xLabel.size() / 2) * 10), dispHeight - 40, xLabel.c_str(), lineColour, 0, 1, 30, 30);

	//Draw Axis Lines
	visu.draw_line(xMinAxis, yMinAxis, xMinAxis, yMaxAxis, lineColour, 1);
	visu.draw_line(xMinAxis, yMaxAxis, xMaxAxis, yMaxAxis, lineColour, 1);

	//Convert Vector Values to the appropriate pix coordinates
	for (int idx = 0; idx < x.size(); idx++) {

		if (x[idx] > maxX)
			maxX = x[idx] + 5.0f;

		if (y[idx] > maxY)
			maxY = y[idx] + 1.0f;

		x[idx] = (x[idx] - minX) / maxX;
		x[idx] = x[idx] * (xMaxAxis - xMinAxis);

		y[idx] = (y[idx] - minY) / maxY;
		y[idx] = (1 - y[idx]) * (yMaxAxis - yMinAxis);

		if (idx > 1) { // Draw line
			visu.draw_line(x[idx - 1] + xMinAxis, y[idx - 1] + yMinAxis, x[idx] + xMinAxis, y[idx] + yMinAxis, lineColour, 1);
		}
	}

	visu.display(disp);

	//Alternatives, simpler with less control
	//Draw each point http://www.cplusplus.com/forum/general/82584/
	// locks after it draws https://stackoverflow.com/questions/39414084/plotting-a-vector-in-c-with-cimg
}

//////////////////////////////////////////////////////
// - Step 6 -
// Predict Results, Save and Load Data set and Matrices
//////////////////////////////////////////////////////

Eigen::MatrixXd Predict(Eigen::MatrixXd weights, Eigen::VectorXd b, Eigen::MatrixXd X) {

	double threshold = 0.5f;

	Eigen::MatrixXd preds = ForwardPropagation(weights, b, X);

	preds = (preds.array() > threshold).select(1, preds);
	preds = (preds.array() <= threshold).select(0, preds);

	return preds;
}

double CalcError(Eigen::MatrixXd real_Y, Eigen::MatrixXd pred_Y) {

	if (real_Y.cols() != pred_Y.cols()) {
		std::cout << "Real Y and Pred Y need be same dimensions \n";
		return -1.0f;
	}

	double error = (real_Y - pred_Y).cwiseAbs().sum();
	error /= real_Y.cols();

	return error;
}

template<typename T>
void Serialise(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, std::string fileName = "matrix") {

	std::fstream writeFile;
	writeFile.open(fileName, std::ios::binary | std::ios::out);

	if (writeFile.is_open())
	{
		int rows, cols;
		rows = m.rows();
		cols = m.cols();

		writeFile.write((const char *)&(rows), sizeof(int));
		writeFile.write((const char *)&(cols), sizeof(int));

		writeFile.write((const char *)(m.data()), sizeof(T) * rows * cols);

		writeFile.close();
	}
}

template<typename T>
void Deserialise(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, std::string fileName = "matrix.eigm") {
	std::fstream readFile;
	readFile.open(fileName, std::ios::binary | std::ios::in);
	if (readFile.is_open())
	{
		int rows, cols;
		readFile.read((char*)&rows, sizeof(int));
		readFile.read((char*)&cols, sizeof(int));

		m.resize(rows, cols);

		readFile.read((char*)(m.data()), sizeof(T) * rows * cols);

		readFile.close();

	}
}

int main() {

	std::string trainFolderDogs = "../Img/Train/Dogs";
	std::string trainFolderNotDogs = "../Img/Train/Not Dogs";
	std::string devFolderDogs = "../Img/Dev/Dogs";
	std::string devFolderNotDogs = "../Img/Dev/Not Dogs";
	std::string testFolder = "../Img/Test";

	bool loadDBFromFiles = false;
	int imgRescaleValue = 64;

	//Initialisation Load Dataset -------------------
	Eigen::MatrixXd TrainingSamples;
	Eigen::MatrixXi TrainingSamplesClasses;
	//Class 1 - Dogs
	//Class 2 - Not Dogs
	if (loadDBFromFiles) {
		Deserialise(TrainingSamples, "../Cereal Database/TrainingSamples.eigm");
		Deserialise(TrainingSamplesClasses, "../Cereal Database/TrainingSamplesClasses.eigm");
	}
	else {
		LoadSet(trainFolderDogs, trainFolderNotDogs, imgRescaleValue, TrainingSamples, TrainingSamplesClasses);
		//Save Eigen Mat Files
		Serialise(TrainingSamples, "../Cereal Database/TrainingSamples.eigm");
		Serialise(TrainingSamplesClasses, "../Cereal Database/TrainingSamplesClasses.eigm");
	}

	Eigen::MatrixXd DevSamples;
	Eigen::MatrixXi DevSamplesClasses;
	//Class 1 - Dogs
	//Class 2 - Not Dogs
	if (loadDBFromFiles) {
		Deserialise(DevSamples, "../Cereal Database/DevSamples.eigm");
		Deserialise(DevSamplesClasses, "../Cereal Database/DevSamplesClasses.eigm");
	}
	else {
		LoadSet(devFolderDogs, devFolderNotDogs, imgRescaleValue, DevSamples, DevSamplesClasses);
		//Save Eigen Mat Files
		Serialise(DevSamples, "../Cereal Database/DevSamples.eigm");
		Serialise(DevSamplesClasses, "../Cereal Database/DevSamplesClasses.eigm");
	}

	//Initialisation --------------------------------
	Eigen::MatrixXd weights;
	Eigen::VectorXd b;
	InitializeNeuron(imgRescaleValue, weights, b);

	//Single Fwd Prop Step --------------------------
	Eigen::MatrixXd preds = ForwardPropagation(weights, b, TrainingSamples);

	//Calc cost after Fwd Prop ----------------------
	double cost = CalculateCost(preds, TrainingSamplesClasses.cast <double>());

	//Single Back Prop Step -------------------------
	Eigen::MatrixXd dw;
	Eigen::MatrixXd db;
	BackwardPropagation(TrainingSamples, preds, TrainingSamplesClasses.cast <double>(), dw, db);

	//Train with Gradient Descent -------------------
	InitializeNeuron(imgRescaleValue, weights, b);
	GradientDescent(TrainingSamples, TrainingSamplesClasses, weights, b, 150, 0.001, true);

	preds = ForwardPropagation(weights, b, TrainingSamples);
	cost = CalculateCost(preds, TrainingSamplesClasses.cast <double>());

	std::cout << "Final Training Cost: " << cost << "\n";
	preds = Predict(weights, b, TrainingSamples);
	std::cout << "Train set with Gradient Descent Accuracy: " << 100 - CalcError(TrainingSamplesClasses.cast <double>(), preds) * 100 << "\n";
	preds = Predict(weights, b, DevSamples);
	std::cout << "Dev set with Gradient Descent Accuracy: " << 100 - CalcError(DevSamplesClasses.cast <double>(), preds) * 100 << "\n";

	std::cout << "-------------------------------------------------------\n";

	//Train with Batch Gradient Descent -------------
	Eigen::MatrixXd weights_Batch;
	Eigen::VectorXd b_Batch;
	InitializeNeuron(imgRescaleValue, weights_Batch, b_Batch);
	BatchGradientDescent(TrainingSamples, TrainingSamplesClasses, weights_Batch, b_Batch, 32, 150, 0.001, true);

	preds = ForwardPropagation(weights_Batch, b_Batch, TrainingSamples);
	cost = CalculateCost(preds, TrainingSamplesClasses.cast <double>());

	std::cout << "Final Training Cost: " << cost << "\n";
	preds = Predict(weights_Batch, b_Batch, TrainingSamples);
	std::cout << "Train set with Batch Gradient Descent Accuracy: " << 100 - CalcError(TrainingSamplesClasses.cast <double>(), preds) * 100 << "\n";
	preds = Predict(weights_Batch, b_Batch, DevSamples);
	std::cout << "Dev set with Batch Gradient Descent Accuracy: " << 100 - CalcError(DevSamplesClasses.cast <double>(), preds) * 100 << "\n";

	std::cout << "-------------------------------------------------------\n";


	//Eigen Hello World---------------------------------
	//MatrixXd m(2, 2);
	//m(0, 0) = 3;
	//m(1, 0) = 2.5;
	//m(0, 1) = -1;
	//m(1, 1) = m(1, 0) + m(0, 1);
	//std::cout << m << std::endl;

	////CImg Hello World--------------------------------
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

	//Draw Plot Hello World------------------------------	
	//CImgDisplay main_disp(500, 400, "Cost Plot"); // display it

	//Eigen::VectorXd xVals(6);
	//xVals << 1, 2, 3, 4, 5, 6;
	//Eigen::VectorXd yVals(6);
	//yVals << 4, 2, 1, 0.5, 0.25, 0.125;
	//std::vector<double> xVec(xVals.data(), xVals.data() + xVals.size());
	//std::vector<double> yVec(yVals.data(), yVals.data() + yVals.size());

	//drawPlot(main_disp, xVals, yVals, 0.0f, 15.0f, 0.0f, 1.0f, "Iterations", "Cost");

	return 0;
}