#!/bin/sh

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
ls VOCdevkit/VOC2007/Annotations | awk 'BEGIN{printf("{\"names\": [\n")} {printf("\"%s\",\n", $1)} END{printf("]}\n")}' > annotations.json
