import argparse
import json
import logging
import os
import pprint
import time
import random
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
from jobspy import scrape_jobs

from jobhunter import config

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

pp = pprint.PrettyPrinter(indent=4)
logging.basicConfig(level=config.LOGGING_LEVEL)

def search_jobs(
    search_term: str, 
    page: int = 1, 
    num_pages: int = 1,
    country: str = "us",
    date_posted: str = "all",
    delay: float = 1.0
) -> List[Dict]:
    """
    Scrape live job postings directly from LinkedIn, Indeed, Glassdoor, and ZipRecruiter using python-jobspy (100% Free).
    """
    logging.info(f"Scraping jobs for '{search_term}' (country: {country}) using python-jobspy (100% Free)...")
    
    # Map date_posted parameter
    hours_old = None
    if date_posted == "today":
        hours_old = 24
    elif date_posted in ["3days", "week"]:
        hours_old = 168
    elif date_posted == "month":
        hours_old = 720

    # Clean country and map to jobspy enum and default location string
    country_map = {
        "ind": ("india", "India"),
        "in": ("india", "India"),
        "us": ("usa", "United States"),
        "usa": ("usa", "United States"),
        "uk": ("uk", "United Kingdom"),
        "lon": ("uk", "London, UK"),
        "au": ("australia", "Australia"),
        "de": ("germany", "Germany"),
        "fra": ("france", "France"),
        "es": ("spain", "Spain"),
        "it": ("italy", "Italy"),
    }
    
    country_clean = country.lower().strip()
    jobspy_country, loc_str = country_map.get(country_clean, ("usa", "United States"))

    # If search_term contains "jobs in", extract the title and location
    clean_search_term = search_term
    if " jobs in " in search_term.lower():
        parts = search_term.lower().split(" jobs in ")
        clean_search_term = parts[0].strip()
        loc_str = parts[1].strip()

    logging.info(f"Final JobSpy params -> search_term: '{clean_search_term}', location: '{loc_str}', country_search: '{jobspy_country}'")

    try:
        # Scrape from multiple job boards concurrently
        jobs_df = scrape_jobs(
            site_name=["linkedin", "indeed", "glassdoor", "ziprecruiter"],
            search_term=clean_search_term,
            location=loc_str,
            results_wanted=num_pages * 15,
            hours_old=hours_old,
            country_search=jobspy_country
        )
    except Exception as e:
        logging.error(f"Error scraping jobs: {e}", exc_info=True)
        jobs_df = pd.DataFrame()

    if jobs_df.empty:
        logging.warning(f"No jobs found for '{search_term}'.")
        raise ValueError(f"No jobs found for search term: '{search_term}'")

    # Convert DataFrame to list of dicts matching JSearch raw schema
    formatted_jobs = []
    for _, row in jobs_df.iterrows():
        job_url = str(row.get("job_url_direct") if pd.notna(row.get("job_url_direct")) and row.get("job_url_direct") else row.get("job_url", ""))
        
        job_dict = {
            "primary_key": str(row.get("id", "")),
            "job_id": str(row.get("id", "")),
            "job_posted_at_datetime_utc": str(row.get("date_posted", "")),
            "job_title": str(row.get("title", "")),
            "employer_name": str(row.get("company", "")),
            "job_apply_link": job_url,
            "employer_logo": str(row.get("company_logo", "")),
            "employer_website": str(row.get("company_url", "")),
            "employer_company_type": "",
            "job_employment_type": str(row.get("job_type", "")),
            "job_is_remote": bool(row.get("is_remote", False)),
            "job_offer_expiration_datetime_utc": "",
            "job_min_salary": float(row.get("min_amount", 0)) if pd.notna(row.get("min_amount")) else None,
            "job_max_salary": float(row.get("max_amount", 0)) if pd.notna(row.get("max_amount")) else None,
            "job_salary_currency": str(row.get("currency", "USD")),
            "job_salary_period": str(row.get("interval", "year")),
            "job_benefits": [],
            "job_city": str(row.get("location", "")),
            "job_state": "",
            "job_country": country,
            "apply_options": [],
            "job_required_skills": [],
            "job_required_experience": {},
            "job_required_education": {},
            "job_description": str(row.get("description", "")),
            "job_highlights": {},
        }
        formatted_jobs.append(job_dict)

    logging.info(f"Successfully scraped {len(formatted_jobs)} jobs.")
    return formatted_jobs

def main(search_term, page=1, num_pages=config.PAGES, country="us", date_posted="all"):
    return search_jobs(
        search_term=search_term, 
        page=page,
        num_pages=num_pages,
        country=country,
        date_posted=date_posted
    )

def entrypoint():
    parser = argparse.ArgumentParser(description="This searches for jobs")
    parser.add_argument("search", type=str, metavar="search", help="the term to search for, like job title")
    parser.add_argument("--page", type=int, default=1, metavar="page", help="the page of results, page 1, 2, 3,...etc.")
    parser.add_argument("--num-pages", type=int, default=config.PAGES, help=f"number of pages to fetch (default: {config.PAGES})")
    parser.add_argument("--country", type=str, default="us", help="country code for job search (e.g., 'us', 'uk')")
    parser.add_argument("--date-posted", type=str, default="all", help="time frame for job posting (e.g., 'all', 'today', 'week', 'month')")

    args = parser.parse_args()
    result = main(
        search_term=args.search, 
        page=args.page,
        num_pages=args.num_pages,
        country=args.country,
        date_posted=args.date_posted
    )
    pp.pprint(result)

if __name__ == "__main__":
    entrypoint()
