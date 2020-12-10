/*
 * ch2model.h
 *
 *  Created on: Dec 9, 2020
 *      Author: zf
 */

#ifndef CH2_CH2MODEL_H_
#define CH2_CH2MODEL_H_


class Ch2Model {
protected:
	const int actionNum;

public:
	Ch2Model(int num);
	virtual ~Ch2Model();

	virtual int getAction() = 0;

	virtual void update(int action, float reward) = 0;
};




#endif /* CH2_CH2MODEL_H_ */
