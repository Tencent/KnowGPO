import pathlib
import re
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[2]
PATTERN=re.compile(r'[A-Za-z0-9]{20,}@[0-9]{4}|Bearer\s+(?!REPLACE_WITH)[A-Za-z0-9_-]{14,}|(?i:api_key)\s*=\s*["\x27][a-f0-9]{32}["\x27]')

class Inventory(unittest.TestCase):
    def test_no_hard_coded_credentials_in_release_files(self):
        flagged=[]
        for p in list((ROOT/'verl').rglob('*.py')) + list((ROOT/'verl/trainer/config').rglob('*.yaml')) + [ROOT/'scripts/train_knowgpo.sh', ROOT/'pyproject.toml']:
            for n,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                if PATTERN.search(line):flagged.append(f'{p.relative_to(ROOT)}:{n}')
        self.assertEqual(flagged,[])

if __name__=='__main__':unittest.main()
