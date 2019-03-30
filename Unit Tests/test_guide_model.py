from guide_berkedia import wrapper
import pytest
import urllib.request
import urllib
@pytest.mark.parametrize("query,expected", [
    ("how to create a shared mailbox",[714,715]),
    ("Unable to launch apps from RDS",[1053, 376]),
    ("User reported auto log off/auto reboot issue",[102, 102]),
    ("High memory utilisation",[1049, 1050]),
    ("Flicker issue thin client",[1053, 1049]),
    ("Unable to access salesforce",[218, 218]),
    ("Unable to login to salesforce",[218, 218]),
    ("Reserving a conference room",[336, 338]),
    ("Update phone number",[194, 195]),
    ("Flashing tpm firmware on windows 10",[1166, 1168]),
    ("Windows 10 login error",[669, 670]),
    ("Outlook password reset",[105, 106]),
    ("Remote access berkedia",[133, 134]),
    ("Access rights in exchange",[309, 310]),
    ("Unable to move mail to the archive",[303, 303])

])

def test_guide(query,expected):
    (links,page_num,confidences)= wrapper(query,1)
    page_num=page_num[0:2]
    assert  page_num == expected

@pytest.mark.parametrize("model",["guide","ticket"])
def test_query_not_string(model):
    data = '{"query":1}'.encode("utf-8")
    url = 'http://localhost:5005/IR/'+model
    req = urllib.request.Request(url, data, {'Content-Type': 'application/json'})
    with pytest.raises(urllib.error.HTTPError): 
        urllib.request.urlopen(req)
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        assert e.code==415

@pytest.mark.parametrize("model",["guide","ticket"])
def test_query_empty(model):
    data = '{"query":}'.encode("utf-8")
    url = 'http://localhost:5005/IR/'+model
    req = urllib.request.Request(url, data, {'Content-Type': 'application/json'})
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(req)
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        assert e.code==400

