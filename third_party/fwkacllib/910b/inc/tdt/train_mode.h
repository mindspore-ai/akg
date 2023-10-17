/**
* @file train_mode.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2018-2019. All Rights Reserved.
*
* This program is used to get or set train mode
*/


#ifndef INC_TDT_TRAIN_MODE_H
#define INC_TDT_TRAIN_MODE_H

enum TrainMode {
    NOFLAG = -1,
    DPFLAG = 0,
    MEFLAG = 1
};

TrainMode GetTrainMode();

void SetTrainMode(TrainMode mode);

#endif
