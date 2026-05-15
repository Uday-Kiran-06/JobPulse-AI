import argparse
import concurrent.futures
import json
import logging
import os
import pprint
import time
import random
import platform
from pathlib import Path

from dotenv import load_dotenv
from jobhunter.FileHandler import FileHandler
from tqdm import tqdm
from jobhunter import config
from jobhunter.search_jobs import search_jobs

# change current director to location of this file
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

file_handler = FileHandler(
    raw_path=config.RAW_DATA_PATH, processed_path=config.PROCESSED_DATA_PATH
)

load_dotenv()
load_dotenv("../.env")
load_dotenv("../../.env")

def get_all_jobs(search_term, pages, country="us", date_posted="all"):
    all_jobs = []
    
    pages_to_fetch = pages
    for page_index in range(pages_to_fetch):
        page_number_for_api = page_index + 1
        try:
            logging.info(f"Fetching page {page_number_for_api}/{pages} for search term '{search_term}' using python-jobspy")
            jobs = search_jobs(
                search_term=search_term, 
                page=page_number_for_api,
                num_pages=1,
                country=country,
                date_posted=date_posted,
                delay=1.0
            )
            
            if jobs:
                logging.debug(f"Received {len(jobs)} jobs from scraper")
                all_jobs.extend(jobs)
                for job in jobs:
                    try:
                        file_handler.save_data(
                            data=job,
                            source="jobs",
                            sink=file_handler.raw_path,
                        )
                    except Exception as e:
                        logging.error(f"Failed to save job data: {str(e)}")
        except ValueError as e:
            logging.warning(f"{str(e)}")
            break 
        except Exception as e:
            logging.error(f"An error occurred while fetching jobs: {str(e)}")
            break 
            
    if not all_jobs:
        error_msg = f"No jobs found for search term: '{search_term}' across all {pages} pages"
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    logging.info(f"Total jobs found for '{search_term}': {len(all_jobs)}")
    return all_jobs


def extract(POSITIONS, country="us", date_posted="all", location=""):
    logging.info("=== ENVIRONMENT DIAGNOSTICS ===")
    logging.info(f"Platform: {platform.platform()}")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"RAW_DATA_PATH: {config.RAW_DATA_PATH}")
    logging.info(f"PROCESSED_DATA_PATH: {config.PROCESSED_DATA_PATH}")
    logging.info("=== END DIAGNOSTICS ===")
    
    try:
        config.RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
        config.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating data folders: {e}")
        raise

    try:
        file_handler = FileHandler()
        file_handler.create_data_folders_if_not_exists()
        
        if not POSITIONS or not isinstance(POSITIONS, list) or len(POSITIONS) == 0:
            error_msg = "No positions provided for job search"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        logging.info(f"Scraper Configuration: Using country='{country}', date_posted='{date_posted}'")
        
        try:
            positions = POSITIONS
            total_jobs_found = 0

            if not location:
                country_locations = {
                    "ind": "India",
                    "in": "India",
                    "us": "United States",
                    "uk": "United Kingdom",
                    "lon": "London, UK",
                    "ca": "Canada",
                    "au": "Australia",
                    "de": "Germany",
                    "fr": "France",
                    "fra": "France",
                    "es": "Spain",
                    "it": "Italy"
                }
                location = country_locations.get(country.lower(), country)
            
            search_results = {}
            
            for position in positions:
                try:
                    if "in " + location.lower() not in position.lower() and " jobs in " not in position.lower():
                        search_term = f"{position} jobs in {location}"
                    else:
                        search_term = position
                    
                    try:
                        jobs = get_all_jobs(
                            search_term=search_term,
                            pages=config.PAGES,
                            country=country,
                            date_posted=date_posted,
                        )
                        
                        job_count = len(jobs)
                        total_jobs_found += job_count
                        search_results[position] = job_count
                        logging.info(f"Found {job_count} jobs for position '{position}'")
                        
                    except ValueError as e:
                        logging.warning(f"Initial search failed: {str(e)}")
                        logging.info("Attempting search with more general terms...")
                        
                        words = position.lower().split()
                        qualifiers = ['senior', 'principal', 'lead', 'staff', 'head', 'chief', 'vp', 'vice president', 'director']
                        position_without_qualifiers = ' '.join([w for w in words if w.lower() not in qualifiers])
                        
                        general_terms = []
                        if position_without_qualifiers and position_without_qualifiers != position:
                            general_terms.append(position_without_qualifiers)
                        
                        domains = ['machine learning', 'data science', 'software engineering', 'ai', 'developer', 'engineer']
                        for domain in domains:
                            if domain in position.lower():
                                general_terms.append(domain)
                                break
                        
                        if not general_terms:
                            general_terms = ['software engineer', 'developer', 'engineer']
                        
                        for term in general_terms:
                            try:
                                fallback_search_term = f"{term} jobs in {location}"
                                fallback_jobs = get_all_jobs(
                                    search_term=fallback_search_term,
                                    pages=config.PAGES,
                                    country=country,
                                    date_posted=date_posted,
                                )
                                if fallback_jobs:
                                    total_jobs_found += len(fallback_jobs)
                                    search_results[position] = len(fallback_jobs)
                                    break
                            except ValueError:
                                continue
                            except Exception as fallback_err:
                                continue
                except Exception as e:
                    search_results[position] = 0
                    continue

            logging.info(f"Total jobs found across all positions: {total_jobs_found}")
            return total_jobs_found

        except Exception as e:
            logging.error(f"Error in extract: {e}")
            return 0
    except Exception as e:
        logging.error(f"Error in extract: {e}")
        return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Job Extraction")
    parser.add_argument("positions", metavar="POSITIONS", type=str, nargs="+", help="List of positions to extract jobs for")
    parser.add_argument("--country", type=str, default="us", help="Country code for job search")
    parser.add_argument("--date-posted", type=str, default="all", help="Time frame for job posting")
    parser.add_argument("--location", type=str, default="", help="Location to search for jobs")
    args = parser.parse_args()
    extract(args.positions, args.country, args.date_posted, args.location)
