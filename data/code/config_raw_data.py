import re
import os
import sys
from pathlib import Path
from deep_translator import GoogleTranslator

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "raw"

def _get_supported_codes():
    """Fetches supported codes from Google Translate."""
    try:
        return set(GoogleTranslator().get_supported_languages(as_dict=True).values())
    except Exception as e:
        return None

def _get_file_action(lang_code, supported_codes):
    """
    Decides the action for a given language code.
    Returns: ('KEEP', code), ('RENAME', new_code), or ('DELETE', None)
    """
    if supported_codes is None:
        return 'KEEP', lang_code

    correction_map = {
        "pt-BR": "pt", "pt-PT": "pt",
        "en-US": "en", "en-GB": "en",
        "es-MX": "es", "es-ES": "es",
        "zh": "zh-CN"
    }

    # 1. Valid as-is
    if lang_code in supported_codes:
        return 'KEEP', lang_code
    
    # 2. Manual Correction
    if lang_code in correction_map:
        new_code = correction_map[lang_code]
        if new_code in supported_codes:
            return 'RENAME', new_code
    
    # 3. Generic Fallback (de-DE -> de)
    if '-' in lang_code:
        base = lang_code.split('-')[0]
        if base in supported_codes:
            return 'RENAME', base
            
    # 4. Invalid
    return 'DELETE', None

def _generate_clean_list():
    """Generates the list of languages as if the folder was clean."""
    final_langs = set()
    filename_pattern = re.compile(r"dev\.en-(.+)\.tsv")
    
    if not RAW_DATA_DIR.exists(): return []

    files = list(RAW_DATA_DIR.glob("*.tsv"))
    supported = _get_supported_codes()

    for file_path in files:
        match = filename_pattern.match(file_path.name)
        if match:
            raw_code = match.group(1)
            action, new_code = _get_file_action(raw_code, supported)
            
            if action == 'KEEP':
                final_langs.add(raw_code)
            elif action == 'RENAME':
                final_langs.add(new_code)

    return sorted(list(final_langs))

# EXPORT THIS VARIABLE (Imported by other files)
LANGUAGES = _generate_clean_list()


def run_cleanup_process(execute_changes=False):
    """
    Scans files and prints a report. 
    If execute_changes is True, it also performs the renames/deletes.
    """
    mode = "ACTION MODE (Modifying files)" if execute_changes else "DRY RUN (Report only)"
    print(f"\n=== LANGUAGE CLEANUP TOOL: {mode} ===")
    
    filename_pattern = re.compile(r"dev\.en-(.+)\.tsv")
    supported = _get_supported_codes()
    
    if supported is None:
        print("Error: Google API offline. Cannot perform cleanup safely.")
        return

    files = list(RAW_DATA_DIR.glob("*.tsv"))
    changes_made = False

    print(f"{'ACTION':<10} | {'FILE / INFO':<40}")
    print("-" * 60)

    for file_path in files:
        match = filename_pattern.match(file_path.name)
        if not match: continue 

        raw_code = match.group(1)
        action, target_code = _get_file_action(raw_code, supported)

        if action == 'DELETE':
            print(f"{'DELETE':<10} | {file_path.name} (Invalid: {raw_code})")
            if execute_changes:
                try:
                    os.remove(file_path)
                    print(f"{' ':<10} | -> DELETED SUCCESS")
                    changes_made = True
                except OSError as e:
                    print(f"{' ':<10} | -> ERROR: {e}")

        elif action == 'RENAME':
            new_name = f"dev.en-{target_code}.tsv"
            new_path = RAW_DATA_DIR / new_name
            print(f"{'RENAME':<10} | {file_path.name} -> {new_name}")
            
            if execute_changes:
                if new_path.exists():
                    print(f"{' ':<10} | -> SKIP: Target exists")
                else:
                    try:
                        file_path.rename(new_path)
                        print(f"{' ':<10} | -> RENAMED SUCCESS")
                        changes_made = True
                    except OSError as e:
                        print(f"{' ':<10} | -> ERROR: {e}")

        else:
            # Just pass, current file has the correct format
            pass

    print("-" * 60)
    if not execute_changes:
        print("To apply these changes, run: python config_raw_data.py --clean")
    elif changes_made:
        print("Cleanup complete. File names updated.")
    else:
        print("No changes were necessary.")
    print("============================================\n")


if __name__ == "__main__":
    if "--clean" in sys.argv:
        run_cleanup_process(execute_changes=True)
    else:
        run_cleanup_process(execute_changes=False)