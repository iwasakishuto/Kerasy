"""
zinc: movebank: https://www.movebank.org/cms/movebank-main
~~~
> Welcome to ZINC, a free database of commercially-available compounds for
  virtual screening. ZINC contains over 230 million purchasable compounds in
  ready-to-dock, 3D formats. ZINC also contains over 750 million purchasable
  compounds you can search for analogs in under a minute.
~~~
"""
import json
from six.moves.urllib import request, parse

from . import get_file

def getSMILES(identifiler):
    url = "https://zinc.docking.org/substances/resolved/"
    if isinstance(identifiler, int):
        identifiler = f"ZINC{'12':>012}"

    post_form_data = parse.urlencode({
        "paste"         : identifiler,
        'upload'        : None,
        'identifiers'   : 'y',
        'structures'    : 'y',
        'names'         : 'y',
        'retired'       : 'y',
        'charges'       : 'y',
        'scaffolds'     : 'y',
        'multiple'      : 'y',
        'output_format' : 'table',
    })
    post_form_headers = {
        'Accept': 'application/json',
    }
    post_req = request.Request(url=url,
                               data=post_form_data.encode(),
                               headers=post_form_headers,
                               method='POST')
    with request.urlopen(post_req) as res:
        body = json.loads(res.read())
    smiles = body[0].get("smiles")
    return smiles
