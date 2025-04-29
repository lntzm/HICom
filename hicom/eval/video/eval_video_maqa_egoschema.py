import os
import json
import requests
import argparse


def convert_csv2json(ori_file, des_file=None):
    with open(ori_file, 'r') as f:
        data = f.readlines()

    json_data = {}
    for line in data[1:]:
        uid, answer = line.split(', ')
        uid = uid.strip()
        answer = int(answer.strip())
        json_data[uid] = answer

    if des_file is not None:
        with open(des_file, 'w') as f:
            json.dump(json_data, f)
    
    return json_data


def send_post_request(json_file):
    """
    Sends a POST request to the specified URL with the given JSON file.

    Parameters:
    - json_file (str): Path to the JSON file to be used in the request body.

    Returns:
    - Response object containing server's response.
    """

    url = "https://validation-server.onrender.com/api/upload/"
    headers = {
        "Content-Type": "application/json"
    }

    # with open(json_file, 'r') as f:
    #     data = json.load(f)
    data = json_file

    response = requests.post(url, headers=headers, json=data)
    
    return response

def main():
    """
    Main function that parses command-line arguments and sends a POST request.
    """

    parser = argparse.ArgumentParser(description="Send a POST request with a CSV file.")
    parser.add_argument("--file", required=True, help="Path to the CSV file to be sent with the request.")
    parser.add_argument("--des_file", type=str, default=None)
    args = parser.parse_args()
    json_data =  convert_csv2json(args.file, args.des_file)
    response = send_post_request(json_data)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content:\n{response.text}")

if __name__ == "__main__":
    main()