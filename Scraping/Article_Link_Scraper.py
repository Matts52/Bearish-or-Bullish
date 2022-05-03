from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
import csv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time



#Global dictionary of month codes
m_codes = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}


def scrape_links_for_day(year, month, day, start_new=False, save_place="WSJ_Art_Links.csv"):
    '''
    Scrape the directory of WSJ journal articles that were published in a given day and write to specified csv file
    
    @param year: integer specifying the year to be scraped
    @param month: integer specifying the month to be scraped
    @param day: integer specifying the day to be scraped
    @param start_new: boolean which is true if we are starting a fresh csv file
    @param save_place: string containgin the file name to by output to > /WSJ_Art_Links.csv by default
    @return: None
    '''
    
    # if we are writing to a fresh csv file, write a header row
    if start_new:
        header = ['year', 'month', 'day', 'link', 'title', 'author']
        with open(save_place, 'w', newline='\n') as f:
            write = csv.writer(f)
            write.writerow(header)

    #grab the corresponding issue, adding a leading zero if necessary
    addInDay = ''
    if day <= 9: addInDay = '0'
    
    addInMonth = ''
    if month <= 9: addInMonth = '0'
    
    # build and grab the url html with the specified information (Retracted for site privacy)
    url = ""
    driver.get(url)

    time.sleep(5)
    html = driver.page_source
    # convert to bs4 object for easier parsing
    soup = BeautifulSoup(html, 'html.parser')

    # grab all links and titles on the current days page
    links = []
    titles = []
    for a in soup.find_all("a", {"id": "citationDocTitleLink"}):
        links.append(a['href'])
        titles.append(a['title'])

    authors = []
    i = 0
    for e in soup.find_all("span", {"class": "titleAuthorETC"}):
        if i % 2 == 0:
            newA = e.text.split(';')[0]
            if newA != '' and newA[-1] == '\n':
                authors.append(newA[:-1])
            else:
                authors.append(newA)
        i += 1

    # format into csv ready rows
    newIn = []
    for i in range(0,len(links)):
        try:
            newIn.append([year, month, day, links[i], titles[i], authors[i]])
        except:
            pass

    #save the current day's content to the save destination
    with open(save_place, 'a', newline='\n') as f:
        write = csv.writer(f)
        for r in newIn:
            try:
                write.writerow(r)
            except:
                pass
    



#Initializing the webdriver, headless so it runs in background
options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(options=options)
driver.set_window_size(1120, 1000)

time.sleep(5)

#customize time frame
start_yr, end_yr, start_mn, end_mn, start_dy, end_dy = 2008,2008,1,12,1,32
for year in range(start_yr, end_yr+1, 2):
    check = True
    for month in range(start_mn, end_mn+1):
        for day in range(start_dy, end_dy+1):
            #run the scraper
            start = time.time()
            scrape_links_for_day(year, month, day, start_new=check, save_place="WSJ_Art_Links_"+str(year)+".csv")
            end = time.time()
            print(year, month, day)
            print(end - start)
            check = False

#quit the chrome driver
driver.quit()

