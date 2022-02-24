import requests
import os
import json
import csv
import time

with open('keys.txt', 'r') as tfile:
    consumer_key = tfile.readline().strip('\n')
    consumer_secret = tfile.readline().strip('\n')
    access_token = tfile.readline().strip('\n')
    access_secret = tfile.readline().strip('\n')
    bearer_token = tfile.readline().strip('\n')

def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FilteredStreamPython"
    return r

def get_rules():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", auth=bearer_oauth
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))
    return response.json()

def delete_all_rules(rules):
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    print(json.dumps(response.json()))

def set_rules(delete):
    sample_rules = [
        {"value": "context:123.1220701888179359745", "tag": "covid-19"},
        
    ]
    payload = {"add": sample_rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))
    
    def get_params():
        return {"tweet.fields": "lang,created_at",
            "expansions": "author_id",
           }
        
filename = "tweets"

def get_stream(set):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream", auth=bearer_oauth, stream=True, params=get_params())
   

    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    

    file = open(filename+".csv", 'a')
    writer = csv.writer(file)
    writer.writerow(["tweet", "tweet URL", "author id", "created at"])

    for response_line in response.iter_lines():
        if response_line:
            decoded_line = response_line.decode('utf-8')
            json_response = json.loads(decoded_line)
            
            if json_response['data']:
                if json_response['data']['lang'] != 'pt':
                    continue
            
                tweetText = json_response['data']['text']
                authorId = json_response['data']['author_id']
                tweetId = json_response['data']['id']
                createdAt = json_response['data']['created_at']

                linkTweet = "https://twitter.com/user/status/" + str(tweetId)

                writer.writerow([tweetText, linkTweet, authorId, createdAt])
            
    file.close()
    
def main():
    rules = get_rules()
    delete = delete_all_rules(rules)
    set = set_rules(delete)
    
    timeout = 0
    
    while True:
        time.sleep(2**timeout)
        timeout += 1
        get_stream(set)


if __name__ == "__main__":
    main()