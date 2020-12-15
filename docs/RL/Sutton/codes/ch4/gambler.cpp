/*
 * gambler.cpp
 *
 *  Created on: Dec 15, 2020
 *      Author: zf
 */



#include <vector>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>

#include <matplotlibcpp.h>

namespace {
const float Ph = 0.55;
const int Target = 100;
const float StepReward = 0;
const float Gamma = 1;
const float Theta = 1e-4;

std::vector<float> vs(Target + 1, 0);
std::vector<int> as(Target + 1, 1);

template<typename T>
void printVec(const std::vector<T>& data, std::string comment) {
	std::cout << comment << std::endl;
	for (int i = 0; i < data.size(); i ++) {
		std::cout << data[i] << ", ";
//		if ((i + 1) % 10 == 0) {
//			std::cout << std::endl;
//		}
	}
	std::cout << std::endl;
}

void plotDatas(const std::vector<int>& actions, const std::vector<float>& values) {
	matplotlibcpp::clf();

	matplotlibcpp::subplot(1, 2, 1);
	matplotlibcpp::bar(actions);
	matplotlibcpp::subplot(1, 2, 2);
	matplotlibcpp::plot(values);
	matplotlibcpp::pause(1);
}

void initVs() {
	vs[0] = 0;
	vs[Target] = 1;

	srand(time(NULL));
	for (int i = 1; i < Target; i ++) {
		as[i] = rand() % i;
	}

	as[Target] = 0;
}

float getNextValue(int s, int a) {
	return (1 - Ph) * vs[s - a] + Ph * vs[(int)std::min(s + a, Target)];
}

void valueEval() {
	float delta = 0;
	int round = 0;

	do {
		std::cout << "value update " << round ++ << std::endl;

		delta = 0;
		for (int i = Target - 1; i > 0; i --) {
			float origV = vs[i];
			vs[i] = getNextValue(i, as[i]);

			delta = std::max(delta, (float)abs(vs[i] - origV));
		}
	} while (delta > Theta);
}

void policyIter() {
	float delta = 0;
	int round = 0;

	do {
		std::cout << "Policy update " << round ++ << std::endl;

		delta = 0;
		for (int s = 1; s < Target; s ++) {
			float origV = vs[s];
			float maxV = 0;
			int maxA = as[s];
			for (int stake = 0; stake <= std::min(s, Target - s); stake ++) {
				float v = getNextValue(s, stake);
				if (v > maxV) {
					maxV = v;
					maxA = stake;
				}
			}
//			if (abs(vs[s] - origV) > Theta) {
//				std::cout << "value " << s << " updated " << vs[s];
//			}
			vs[s] = maxV;
			as[s] = maxA;
			delta = std::max(delta, (float)abs(vs[s] - origV));
		}
		printVec(vs, "Values");
		std::cout << "delta = " << delta << std::endl;
		plotDatas(as, vs);
	} while (delta > Theta);

	matplotlibcpp::save("./test.jpg");
}

}

int main() {
	initVs();
	policyIter();
}
