SHELL := /bin/bash

.PHONY: prerequisites install

# Small wrapper around pdm to setup the project

prerequisites:
	# Make sure to have conda installed.
	# This will allow you to create a new environment with all the dependencies needed for this project.
	# Conda environments also contain dedicated python interpreters that won't mess up your local python installation.

install:
	# Create a new environment
	conda env create -y -f=environment.yml
	# Initialize conda for the current shell session and activate the environment
	@source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate bridging-the-domain-gap-by-diffusion && pdm install  # Install the project dependencies