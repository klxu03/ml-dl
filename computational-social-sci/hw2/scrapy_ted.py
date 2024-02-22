import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess

class TEDSpider(scrapy.Spider):
    name = "ted_spider"

    def __init__(self, *args, **kwargs):
        super(TEDSpider, self).__init__(*args, **kwargs)
        self.talk_descriptions = {"talkid": [], "description": []}

    def start_requests(self):
        df = pd.read_csv('data/common2.csv')

        for talk_id in df['talkid']:
            url = f'https://www.ted.com/talks/{talk_id}'
            yield scrapy.Request(url=url, callback=self.parse, meta={'talk_id': talk_id})
            print(f"Scraping talk {talk_id}")

    def parse(self, response):
        # Extract the talk description using the class or data-testid
        talk_description = response.css('span[data-testid="talk-description-text"]::text').get()

        self.talk_descriptions["talkid"].append(response.meta["talk_id"]) 
        self.talk_descriptions["description"].append(talk_description) 

    def closed(self, reason):
        print(f"Talk descriptions: {self.talk_descriptions}")
        original_df = pd.read_csv('data/common2.csv')
        descriptions_df = pd.DataFrame.from_dict(self.talk_descriptions)

        merged_df = original_df.merge(descriptions_df, on='talkid', how='left')
        print(f"Done scraping merged_df {merged_df}")
        merged_df.to_csv('data/common2_with_descriptions.csv', index=False)

if __name__ == '__main__':
    process = CrawlerProcess()
    process.crawl(TEDSpider)
    process.start()