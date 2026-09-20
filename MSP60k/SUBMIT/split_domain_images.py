import os
import shutil

def main():
    # Map image prefixes to domain names based on ms_split.py configuration
    # Construction Site: part1 (part1_), youtian (part2_)
    # Market: leqing (part0_), nongmao (part5_), huashu (part6_)
    # Kitchens: nannong (part7_), part3 (part8_), hs (part10_), nfy (part11_), hys (part12_), mclz (part13_)
    # Ski Resort: Ski (part16_)
    # School: sucai (part4_)
    # Outdoors: CleanAHU (part15_), AHU_XSF (part14_), zaxiang (part9_), Agdao (part17_)
    domain_mapping = {
        'part1_': 'Construction Site',
        'part2_': 'Construction Site',
        
        'part0_': 'Market',
        'part5_': 'Market',
        'part6_': 'Market',
        
        'part7_': 'Kitchens',
        'part8_': 'Kitchens',
        'part10_': 'Kitchens',
        'part11_': 'Kitchens',
        'part12_': 'Kitchens',
        'part13_': 'Kitchens',
        
        'part16_': 'Ski Resort',
        
        'part4_': 'School',
        
        'part15_': 'Outdoors',
        'part14_': 'Outdoors',
        'part9_':  'Outdoors',
        'part17_': 'Outdoors'
    }
    
    # Source directory containing all images
    source_dir = 'images'
    # Target directory to store the domain-separated images
    target_dir = 'domain_images'
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    print("Creating target directories...")
    for domain in set(domain_mapping.values()):
        os.makedirs(os.path.join(target_dir, domain), exist_ok=True)
        
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images in '{source_dir}'. Starting to copy...")
    
    unmatched_count = 0
    for img_name in image_files:
        matched = False
        for prefix, domain in domain_mapping.items():
            if img_name.startswith(prefix):
                src_path = os.path.join(source_dir, img_name)
                dst_path = os.path.join(target_dir, domain, img_name)
                
                # Copying files (use shutil.move if you want to move them instead)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)
                
                matched = True
                break
        
        if not matched:
            unmatched_count += 1
            # Uncomment the next line to debug unmatched images
            # print(f"Warning: Could not determine domain for image '{img_name}'")
            
    print("\nFinished processing all images!")
    if unmatched_count > 0:
        print(f"Warning: {unmatched_count} images did not match any domain.")
    else:
        print("All images were successfully matched to a domain.")

if __name__ == '__main__':
    main()
