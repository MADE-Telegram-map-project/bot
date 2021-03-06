from typing import List, Tuple
import requests
import re
from bs4 import BeautifulSoup


def extract_subscribers(channel_name: str):
    '''
    This parser-function extracts available info about channels from their web-pages generated by telegram.
    Using of HTTP can help us avoiding FloodErrors and bans

    return -1 in case of username is not a channel, channel is private or error
    '''
    try:
        r = requests.get("https://t.me/{}".format(channel_name))
        HTML_content = r.content  # store HTML into the variable content
        soup = BeautifulSoup(HTML_content, "html.parser")

        # here we find the right tag with subcsribers
        html_tag_with_subscribers = soup.find("div", {"class": "tgme_page_extra"})
        text_number_of_subscribers = html_tag_with_subscribers.text

        # group here means the found match and we want the first one
        number_of_subscribers_raw = re.search(
            "[\\d\\s]+", text_number_of_subscribers).group(0)
        nos = int(number_of_subscribers_raw.replace(" ", ""))
    except:
        nos = -1
    return nos


def tag2text(tag):
    text = " ".join([x.text for x in tag if len(x.text)])
    return text


def extract_header(soup: BeautifulSoup) -> str:
    try:
        html_tag_with_header = soup.find("div", {"class": "tgme_channel_info_description"})
        header_text = tag2text(html_tag_with_header)
    except:
        header_text = ""
    return header_text


def extract_messages(soup: BeautifulSoup) -> List[str]:
    html_tags_with_messages = soup.find_all("div", {"class": "tgme_widget_message_text"})
    messages = [tag2text(tag) for tag in html_tags_with_messages]
    return messages


def parse_channel_web(channel_name: str) -> Tuple[str, List[str]]:
    try:
        r = requests.get("https://t.me/s/{}".format(channel_name))
        HTML_content = r.content  # store HTML into the variable content
        soup = BeautifulSoup(HTML_content, "html.parser")

        header = extract_header(soup)
        messages = extract_messages(soup)
    except:
        header = ""
        messages = []
        
    return header, messages


if __name__ == "__main__":
    username = "latinapopacanski"
    # username = "fak_tu"
    # username = "kpotoh"
    ns = parse_channel_web(username)
    print(ns)