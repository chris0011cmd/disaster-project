import requests

# change this URL after deployment
API_URL = "http://127.0.0.1:5000/allocate-resources"

data = {
    "building_no_damage": 5,
    "building_minor_damage": 2,
    "building_major_damage": 1,
    "building_total_destruction": 1
}

response = requests.post(API_URL, json=data)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
