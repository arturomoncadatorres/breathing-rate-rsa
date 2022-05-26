# -*- coding: utf-8 -*-
import os
import shutil
import click
import logging
import requests
import zipfile
from io import BytesIO
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('url')
@click.argument('output_filepath', type=click.Path(exists=False))
def main(url='https://physionet.org/static/published-projects/fantasia/fantasia-database-1.0.0.zip', output_filepath='./data/raw'):
    """ 
    Runs data scripts to download the raw data files.
    
    Parameters
    ----------
    url: string
        Physionet URL of the Fantasia database
    output_filepath: string or pathlib.Path
        Location where the raw dataset files will be saved
        
    Returns
    -------
    None. Files are written in output_filepath
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    - Iyengar, Nikhil, et al. "Age-related alterations in the fractal scaling 
    of cardiac interbeat interval dynamics." American Journal of 
    Physiology-Regulatory, Integrative and Comparative Physiology 271.4 
    (1996): R1078-R1084.
    """    
    
    logger = logging.getLogger(__name__)
    logger.info('Downloading raw dataset from ' + url + '...')

    # Downloading dataset by sending request to the URL
    req = requests.get(url)
    
    # Split URL to get the file name
    filename_ext = url.split('/')[-1]  # With .zip extension
    filename = filename_ext[:-4] # Without extension

    logger.info("Download is complete!")

    logger.info("Verifying existance of " + output_filepath + "...")
    if Path(output_filepath).exists():
        logger.info("\t" + output_filepath + " exists already")
    else:
        os.mkdir(output_filepath)
        logger.info("\t" + output_filepath + " did not exist and was created")
    
    # Unzip files and write them to local file system
    logger.info('Unzipping files...')

    zipfile_ = zipfile.ZipFile(BytesIO(req.content))
    zipfile_.extractall(output_filepath)

    for file in (Path(output_filepath)/filename).iterdir():
        
        logger.info('\tUnzipping ' + str(file) + '...')
        shutil.move(file, output_filepath)

    # Delete original directory.
    os.rmdir(Path(output_filepath)/filename)

    logger.info("Unzipping is complete!")
    
    return None



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
