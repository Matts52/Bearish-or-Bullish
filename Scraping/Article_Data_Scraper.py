from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver.common.by import By
import csv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time


def scrape_articles(csv_in, csv_out="WSJ_Art_Strings.csv", start_new=False, l=1):
    '''
    Scrape article text from the provided directory of article links and save them to the specified save location
    
    @param csv_in: file name to read articles links from
    @param csv_out: save location for article text
    @param start_new: Boolean representing whether we are starting a new csv out file
    @param l: count of articles to scrape
    @return: None
    '''

    if start_new:
        header = ['link', 'text']
        with open(csv_out, 'w', newline='\n') as f:
            write = csv.writer(f)
            write.writerow(header)

    #read in lines from the csv
    with open(csv_in, 'r', newline='\n') as f:
        reader = csv.reader(f)
        next(reader)
        
        #open a writer to write output only once so that the program is less system intensive/potentially faster
        with open(csv_out, 'a', newline='\n') as f:
            write = csv.writer(f)
            for i, line in enumerate(reader):
                start = time.time()
                
                #Initializing the webdriver, headless so it runs in background
                options = webdriver.ChromeOptions()
                options.add_argument('headless')
                driver = webdriver.Chrome(options=options)
                driver.set_window_size(1120, 1000)
                
                #get url as well as html and mask account id (Retracted for site privacy)
                url = ""+line[3]
                driver.get(url)
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')

                #grab the main article text
                mainArt = soup.find("div", {"id": "fullTextZone"})

                try:
                    textBlocks = mainArt.find_all("p")
                except:
                    textBlocks = []
            
                tx = ""
                #get all paragraphs from the article and append them
                for j in range(0, len(textBlocks)-2):
                    if '\n' in textBlocks[j]:
                        continue
                    
                    if textBlocks[j].text == "Show less":
                        continue
                    else:
                        tx += " " + textBlocks[j].text

                newIn = [line[3], tx]


                #save the current day's content to the save destination
                #try except for potentially bad characters
                try:
                    write.writerow(newIn)
                except:
                    pass
                        
                end = time.time()
            
                print('\n', round((i/l)*100, 2))
            
                print(end - start)
            

#approximate row count for rough time estimating
lc = 7000

scrape_articles("file_in.csv", start_new=False, l=lc, csv_out="file_out.csv")







