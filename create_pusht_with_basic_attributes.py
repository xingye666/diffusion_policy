def create_pusht_with_basic_attributes(target_path, fill_with_data=None):
    """创建pusht副本并添加基础属性"""
    import zarr
    import numpy as np
    
    # 创建Blosc压缩器（与原数据完全一致）
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
    
    # 创建根组
    store = zarr.open(target_path, mode='w')
    
    print("🏗️ 创建pusht副本（添加基础属性）...")
    
    # 1. 设置根组属性
    store.attrs.update({    
    })
    
    # 2. 创建data组并设置属性
    data_group = store.create_group('data')
    data_group.attrs.update({
    })
    
    # 3. 创建data下的数组
    arrays_config = [
        ('action', (25650, 2), (161, 2), 'float32', "Robot action commands (2D)"),
        ('img', (25650, 96, 96, 3), (161, 96, 96, 3), 'float32', "RGB image observations (96x96)"),
        ('keypoint', (25650, 9, 2), (161, 9, 2), 'float32', "Keypoint positions (9 keypoints, 2D)"),
        ('n_contacts', (25650, 1), (161, 1), 'float32', "Number of contact points"),
        ('state', (25650, 5), (161, 5), 'float32', "Robot state vector (5D)")
    ]
    
    for name, shape, chunks, dtype, description in arrays_config:
        arr = data_group.zeros(name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)
        arr.attrs.update({
        })
        print(f"  ✅ 创建 data/{name}")
    
    # 4. 创建meta组并设置属性
    meta_group = store.create_group('meta')
    meta_group.attrs.update({
    })
    
    # 5. 创建episode_ends数组并设置属性
    episode_ends = meta_group.zeros(
        'episode_ends',
        shape=(206,),
        chunks=(208,),
        dtype='int64',
        compressor=compressor
    )
    episode_ends.attrs.update({
    })
    print(f"  ✅ 创建 meta/episode_ends")
    
    # 6. 如果提供了数据，填充数据
    if fill_with_data is not None:
        print("\n📥 填充数据...")
        
        # 设置数据来源属性
        store.attrs["data_source"] = "Provided by user"
        store.attrs["data_filled"] = True
        
        # 填充数据
        if 'action' in fill_with_data:
            data_group['action'][:] = fill_with_data['action']
            print(f"  ✅ 填充 data/action")
        
        if 'img' in fill_with_data:
            # 分块写入大图像数据
            img_data = fill_with_data['img']
            chunk_size = 161
            for i in range(0, 25650, chunk_size):
                end = min(i + chunk_size, 25650)
                data_group['img'][i:end] = img_data[i:end]
            print(f"  ✅ 填充 data/img (分块写入)")
        
        if 'keypoint' in fill_with_data:
            data_group['keypoint'][:] = fill_with_data['keypoint']
            print(f"  ✅ 填充 data/keypoint")
        
        if 'n_contacts' in fill_with_data:
            data_group['n_contacts'][:] = fill_with_data['n_contacts']
            print(f"  ✅ 填充 data/n_contacts")
        
        if 'state' in fill_with_data:
            data_group['state'][:] = fill_with_data['state']
            print(f"  ✅ 填充 data/state")
        
        if 'episode_ends' in fill_with_data:
            meta_group['episode_ends'][:] = fill_with_data['episode_ends']
            print(f"  ✅ 填充 meta/episode_ends")
    
    print(f"\n🎉 副本创建完成！保存到: {target_path}")
    
    # 验证属性
    print("\n🔍 验证创建的属性:")
    for path in ["", "data", "data/action", "data/img", "data/keypoint", 
                 "data/n_contacts", "data/state", "meta", "meta/episode_ends"]:
        if path == "":
            obj = store
            display_path = "根组"
        else:
            obj = store
            for part in path.split('/'):
                obj = obj[part]
            display_path = path
        
        if obj.attrs:
            print(f"  ✅ {display_path}: {len(obj.attrs)}个属性")
        else:
            print(f"  ⚠️  {display_path}: 无属性")
    
    return store

# 使用示例
if __name__ == '__main__':
    import numpy as np
    
    # 准备测试数据
    your_data = {
        'action': np.random.randn(25650, 2).astype('float32'),
        'img': np.random.rand(25650, 96, 96, 3).astype('float32') * 255,
        'keypoint': np.random.randn(25650, 9, 2).astype('float32'),
        'n_contacts': np.random.rand(25650, 1).astype('float32'),
        'state': np.random.randn(25650, 5).astype('float32'),
        'episode_ends': np.cumsum(np.random.randint(100, 200, 206)).astype('int64')
    }
    
    # 创建带基础属性的副本
    filled_replica_path = "pusht_cchi_v7_replay_with_attrs.zarr"
    store_with_attrs = create_pusht_with_basic_attributes(filled_replica_path, fill_with_data=your_data)
