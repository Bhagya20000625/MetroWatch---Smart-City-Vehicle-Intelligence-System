"""
Province Detector for Sri Lankan License Plates
Extracts and maps province codes from license plate text
"""

import re


class ProvinceDetector:
    
    # Sri Lankan Province Code Mapping
    PROVINCE_MAP = {
        # Western Province
        'WP': 'Western Province',
        
        # Central Province
        'CP': 'Central Province',
        
        # Southern Province
        'SP': 'Southern Province',
        
        # Northern Province
        'NP': 'Northern Province',
        
        # Eastern Province
        'EP': 'Eastern Province',
        
        # North Western Province
        'NW': 'North Western Province',
        
        # North Central Province
        'NC': 'North Central Province',
        
        # Uva Province
        'UP': 'Uva Province',
        
        # Sabaragamuwa Province
        'SG': 'Sabaragamuwa Province',
        
        # Special codes
        'DM': 'Diplomatic',
        'CD': 'Corps Diplomatique',
        'CC': 'Consular Corps',
        'GOV': 'Government',
        'PL': 'Police',
        'SRI': 'Special Registration'
    }
    
    @staticmethod
    def extract_province_code(plate_text):
        if not plate_text:
            return None
        
        # Clean the text
        plate_text = plate_text.upper().strip()
        
        # Remove extra spaces
        plate_text = ' '.join(plate_text.split())
        
        # Sri Lankan format: XX YYYY 9999 or XXX YYYY 9999
        # Extract first part (province code)
        parts = plate_text.split()
        
        if parts:
            province_code = parts[0]
            if re.match(r'^[A-Z]{2,3}$', province_code):
                return province_code

        if len(plate_text) >= 2:
            if len(plate_text) >= 3:
                code_3 = plate_text[:3]
                if code_3 in ProvinceDetector.PROVINCE_MAP:
                    return code_3
            
            code_2 = plate_text[:2]
            if code_2 in ProvinceDetector.PROVINCE_MAP:
                return code_2
        
        return None
    
    @staticmethod
    def get_province_name(province_code):
        if not province_code:
            return "Unknown Province"
        
        province_code = province_code.upper().strip()
        return ProvinceDetector.PROVINCE_MAP.get(province_code, "Unknown Province")
    
    @staticmethod
    def detect_province(plate_text):
        province_code = ProvinceDetector.extract_province_code(plate_text)
        province_name = ProvinceDetector.get_province_name(province_code)
        
        return {
            'plate_text': plate_text,
            'province_code': province_code if province_code else "UNKNOWN",
            'province_name': province_name
        }
    
    @staticmethod
    def get_all_provinces():
        return ProvinceDetector.PROVINCE_MAP.copy()

if __name__ == "__main__":
    # Test cases
    test_plates = [
        "WP CAA 1234",
        "CP ABC 5678",
        "SP XYZ 9012",
        "NW DEF 3456",
        "WPCAA1234", 
        "EP 1234",
        "GOV 123",
        "UNKNOWN PLATE"
    ]
    
    detector = ProvinceDetector()
    
    print("=" * 60)
    print("Sri Lankan Province Detection Test")
    print("=" * 60)
    
    for plate in test_plates:
        result = detector.detect_province(plate)
        print(f"\nPlate Text: {result['plate_text']}")
        print(f"Province Code: {result['province_code']}")
        print(f"Province Name: {result['province_name']}")
        print("-" * 60)
    
    # Show all available provinces
    print("\n" + "=" * 60)
    print("All Available Province Codes:")
    print("=" * 60)
    for code, name in detector.get_all_provinces().items():
        print(f"{code:5} → {name}")
