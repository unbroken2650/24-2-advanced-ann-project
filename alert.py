import os
from datetime import datetime

import requests

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def send_slack_message(attachments):
    response = requests.post(SLACK_WEBHOOK_URL, json={"attachments": attachments})
    if response.status_code != 200:
        raise ValueError(f'Request to Slack returned an error {response.status_code}, the response is: {response.text}')


send_slack_message([{
    "fallback": "코드 수행 완료",
    "color": '#aaaaaa',
    "title": "공지사항",
    "text": "코드 수행 완료",
            "ts": datetime.now().timestamp()
}])
print("코드 수행 완료")
