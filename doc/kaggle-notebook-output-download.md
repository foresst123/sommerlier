# Kaggle Notebook: run and download output

Ghi chu thao tac ngay 2026-09-10, vi sidebar Output cua Kaggle can hover/click dung vi tri moi hien nut tai file.

## Muc tieu

Tao mot notebook Kaggle, chay mot lenh nho sinh file trong `/kaggle/working`, roi tai file `.zip` ve may.

## Dang nhap dung cach

Dung Chrome ben ngoai, khong dung browser nhung trong Codex.

1. Mo `https://myaccount.google.com/` trong Chrome de kiem tra dang o dung tai khoan Google.
2. Tai khoan da dung trong lan nay:
   `lamkdhe180931@fpt.edu.vn`
3. Mo `https://www.kaggle.com/code/new`.
4. Neu Kaggle bao chua dang nhap, bam **Sign In** -> **Sign in with Google**. Chrome da co tai khoan Google nen Kaggle tu quay lai notebook.

## Code test da chay

Dung cell Python nay de tao ket qua:

```python
from pathlib import Path
import zipfile, datetime, random

out_dir = Path('/kaggle/working')
result = out_dir / 'result.txt'
result.write_text(
    'Kaggle test run OK\n'
    f'Time UTC: {datetime.datetime.utcnow().isoformat()}Z\n'
    f'Random number: {random.randint(1000, 9999)}\n',
    encoding='utf-8'
)

zip_path = out_dir / 'result.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(result, arcname='result.txt')

print('created', result)
print('created', zip_path)
```

Chay cell. Khi output hien:

```text
created /kaggle/working/result.txt
created /kaggle/working/result.zip
```

thi file da ton tai trong Kaggle.

## Cach tai file dung tren UI Kaggle

Khong tai bang link trong output cell. Link do co the mo thanh tab `kkb-production.../kaggle/working/result.zip` va bi `404 page not found`.

Lam theo sidebar ben phai:

1. Nhin panel **Output** ben phai.
2. Dua chuot vao hang `/kaggle/working`.
3. Bam mui ten nho ben trai hang `/kaggle/working`.
4. Khi thu muc mo ra, se thay cac file ben duoi, vi du:
   - `result.txt`
   - `result.zip`
5. Dua chuot vao hang `result.zip`.
6. Khi hover dung hang, cac nut an hien ra o mep phai.
7. Bam nut ba cham **More actions for (result.zip)**.
8. Menu hien muc **Download**.
9. Bam **Download**.
10. Kiem tra file trong `C:\Users\aiday\Downloads`.

Lan nay file tai ve thanh cong tai:

```text
C:\Users\aiday\Downloads\result.zip
```

## Ghi nho quan trong

- Nut expand cua `/kaggle/working` co the khong hoat dong neu click theo accessibility ID. Neu vay dung screenshot/toa do, click vao mui ten nho ben trai icon folder.
- Phai hover/click dung hang file `result.zip` thi nut ba cham cua file moi hien ra.
- Dung menu file trong sidebar Output la cach on dinh nhat. Khong dung `FileLink`/URL truc tiep de tai.
