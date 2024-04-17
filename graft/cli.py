#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Commandline interface for GraFT.
"""

import argparse
import os

import skimage.io as io

from graft.main import create_all, create_all_still, generate_default_mask


# Constants for default values
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

    # common arguments for both subparsers
    for subparser in ['timeseries', 'still']:
        parser_sp = subparsers.add_parser(subparser, help=f'Analyze {subparser} image data')
        parser_sp.add_argument('image_path', type=str, help='Path to the input image file')
        parser_sp.add_argument('--mask_path', type=str, help='Optional path to the mask file. If omitted, a default mask is used.')
        parser_sp.add_argument('output_dir', type=str, help='Path to the output directory')
        
        # optional arguments shared between timeseries and still
        parser_sp.add_argument('--size', type=int, default=DEFAULT_SIZE, help='Size parameter')
        parser_sp.add_argument('--eps', type=int, default=DEFAULT_EPS, help='EPS parameter')
        parser_sp.add_argument('--thresh_top', type=float, default=DEFAULT_THRESH_TOP, help='Threshold top parameter')
        parser_sp.add_argument('--sigma', type=float, default=DEFAULT_SIGMA, help='Sigma parameter for tubeness filter width')
        parser_sp.add_argument('--small', type=int, default=DEFAULT_SMALL, help='Small parameter for cluster removal')
        parser_sp.add_argument('--angleA', type=int, default=DEFAULT_ANGLEA, help='AngleA parameter')
        parser_sp.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP, help='Overlap parameter')
        
        if subparser == 'timeseries':
            parser_sp.add_argument('--max_cost', type=int, default=DEFAULT_MAX_COST, help='Max cost parameter for tracking')

    args = parser.parse_args()
    
    # Check if a command (subcommand) has been chosen. If not, display help and exit.
    if args.command is None:
        parser.print_help()
        parser.exit()

    img_o = io.imread(os.path.abspath(args.image_path))

    if args.mask_path:
        maskDraw = io.imread(os.path.abspath(args.mask_path))
    else:
        maskDraw = generate_default_mask(img_o.shape)

    if args.command == 'timeseries':
        create_all(pathsave=os.path.abspath(args.output_dir),
                   img_o=img_o,
                   maskDraw=maskDraw,
                   size=args.size,
                   eps=args.eps,
                   thresh_top=args.thresh_top,
                   sigma=args.sigma,
                   small=args.small,
                   angleA=args.angleA,
                   overlap=args.overlap,
                   max_cost=args.max_cost,
                   name_cell='in silico time')

    elif args.command == 'still':
        create_all_still(pathsave=os.path.abspath(args.output_dir),
                         img_o=img_o,
                         maskDraw=maskDraw,
                         size=args.size,
                         eps=args.eps,
                         thresh_top=args.thresh_top,
                         sigma=args.sigma,
                         small=args.small,
                         angleA=args.angleA,
                         overlap=args.overlap,
                         name_cell='in silico still')

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
