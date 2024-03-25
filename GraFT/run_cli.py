#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced run.py script for GraFT with CLI support.
"""

import argparse
import os
import skimage.io as io

from run import create_all, create_all_still

# Constants for default values as specified in README
DEFAULT_SIZE = 6
DEFAULT_EPS = 200
DEFAULT_THRESH_TOP = 0.5
DEFAULT_SIGMA = 1.0
DEFAULT_SMALL = 50
DEFAULT_ANGLEA = 140
DEFAULT_OVERLAP = 4
DEFAULT_MAX_COST = 100

def main():
    parser = argparse.ArgumentParser(description="GraFT: Graph of Filaments over Time")
    subparsers = parser.add_subparsers(help='commands', dest='command')

    # Subparser for timeseries analysis
    timeseries_parser = subparsers.add_parser('timeseries', help='Analyze timeseries image data')
    timeseries_parser.add_argument('image_path', type=str, help='Path to the input image file')
    timeseries_parser.add_argument('mask_path', type=str, help='Path to the mask file')
    timeseries_parser.add_argument('output_dir', type=str, help='Path to the output directory')
    # Optional arguments
    timeseries_parser.add_argument('--size', type=int, default=DEFAULT_SIZE, help='Size parameter (default: 6)')
    timeseries_parser.add_argument('--eps', type=int, default=DEFAULT_EPS, help='EPS parameter (default: 200)')
    # Add other optional parameters here as needed

    # Subparser for still image analysis
    still_parser = subparsers.add_parser('still', help='Analyze still image data')
    still_parser.add_argument('image_path', type=str, help='Path to the input image file')
    still_parser.add_argument('mask_path', type=str, help='Path to the mask file')
    still_parser.add_argument('output_dir', type=str, help='Path to the output directory')
    # Optional arguments
    still_parser.add_argument('--size', type=int, default=DEFAULT_SIZE, help='Size parameter (default: 6)')
    # Add other optional parameters for still image analysis

    args = parser.parse_args()

    if args.command == 'timeseries':
        img_o = io.imread(args.image_path)
        maskDraw = io.imread(args.mask_path)
        create_all(pathsave=args.output_dir,
                   img_o=img_o,
                   maskDraw=maskDraw,
                   size=args.size,
                   eps=args.eps,
                   thresh_top=DEFAULT_THRESH_TOP,
                   sigma=DEFAULT_SIGMA,
                   small=DEFAULT_SMALL,
                   angleA=DEFAULT_ANGLEA,
                   overlap=DEFAULT_OVERLAP,
                   max_cost=DEFAULT_MAX_COST,
                   name_cell='timeseries_analysis')
    elif args.command == 'still':
        img_o = io.imread(args.image_path)
        maskDraw = io.imread(args.mask_path)
        create_all_still(pathsave=args.output_dir,
                         img_o=img_o,
                         maskDraw=maskDraw,
                         size=args.size,
                         eps=args.eps,
                         thresh_top=DEFAULT_THRESH_TOP,
                         sigma=DEFAULT_SIGMA,
                         small=DEFAULT_SMALL,
                         angleA=DEFAULT_ANGLEA,
                         overlap=DEFAULT_OVERLAP,
                         name_cell='still_image_analysis')
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

