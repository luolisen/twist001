"""Provision a new local key without printing secrets. Never rotate an existing key implicitly."""
from pathlib import Path
import argparse,secrets,json,os
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--new',action='store_true');args=parser.parse_args()
path=root/'config/private/wireless.json'
if path.exists():
 if args.new:raise SystemExit('Existing credentials preserved. Key rotation needs an explicit coordinated firmware/client update.')
 config=json.loads(path.read_text())
else:
 path.parent.mkdir(parents=True,exist_ok=True)
 config={'key_hex':secrets.token_hex(32),'ap_password':secrets.token_urlsafe(16),'sta_ssid':'','sta_password':''}
 path.write_text(json.dumps(config,indent=2)+'\n');path.chmod(0o600)
key=bytes.fromhex(config['key_hex'])
if len(key)!=32 or len(config['ap_password'])<8:raise SystemExit('Invalid configuration')
header='#pragma once\nstatic const unsigned char RADIO_KEY[32] = {'+','.join(str(x) for x in key)+'};\n'
for name,field in [('AP_PASSWORD','ap_password'),('STA_SSID','sta_ssid'),('STA_PASSWORD','sta_password')]:header+='static const char '+name+'[] = '+json.dumps(config[field])+';\n'
target=root/'ESP32_Final/private_config.h';target.write_text(header);target.chmod(0o600)
print('Local provisioning complete; secrets not displayed.')
