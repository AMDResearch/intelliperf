#!/bin/bash

image="apptainer/maestro.sif"
apptainer exec --cleanenv --env OPENAI_API_KEY=$OPENAI_API_KEY $image bash
