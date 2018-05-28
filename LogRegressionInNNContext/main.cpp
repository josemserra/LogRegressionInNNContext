#include "CImg.h"

#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace cimg_library;

int main() {

	//Eigen Hello World
	MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2.5;
	m(0, 1) = -1;
	m(1, 1) = m(1, 0) + m(0, 1);
	std::cout << m << std::endl;

	//CImg Hello World
	CImg<unsigned char> image("../Img/lena.jpg"), visu(500, 400, 1, 3, 0);
	const unsigned char red[] = { 255,0,0 }, green[] = { 0,255,0 }, blue[] = { 0,0,255 };
	CImgDisplay main_disp(image, "Click a point"), draw_disp(visu, "Intensity profile");
	while (!main_disp.is_closed() && !draw_disp.is_closed()) {
		main_disp.wait();
		if (main_disp.button() && main_disp.mouse_y() >= 0) {
			const int y = main_disp.mouse_y();
			visu.fill(0).draw_graph(image.get_crop(0, y, 0, 0, image.width() - 1, y, 0, 0), red, 1, 1, 0, 255, 0);
			visu.draw_graph(image.get_crop(0, y, 0, 1, image.width() - 1, y, 0, 1), green, 1, 1, 0, 255, 0);
			visu.draw_graph(image.get_crop(0, y, 0, 2, image.width() - 1, y, 0, 2), blue, 1, 1, 0, 255, 0).display(draw_disp);
		}
	}

	return 0;
}