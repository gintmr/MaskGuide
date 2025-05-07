def process_performance_metrics(performance_metrics, accumulated_metrics=None):
    """
    处理并累积性能指标
    
    参数:
        performance_metrics: 当前帧的性能指标字典
        accumulated_metrics: 累积的性能指标字典(可选)
        
    返回:
        tuple: (当前帧指标字典, 累积指标字典, 平均指标字典)
    """
    # 初始化累积指标字典
    if accumulated_metrics is None:
        accumulated_metrics = {
            'total_frames': 0,
            'total_time': 0.0,
            'total_gen_time': 0.0,
            'total_pred_time': 0.0,
            'peak_memory': 0,
            'total_gen_memory': 0,
            'total_pred_memory': 0
        }
    
    # 更新累积性能指标
    accumulated_metrics['total_frames'] += 1
    accumulated_metrics['total_time'] += performance_metrics['total_time_seconds']
    accumulated_metrics['total_gen_time'] += performance_metrics['component_times']['generator_time_seconds']
    accumulated_metrics['total_pred_time'] += performance_metrics['component_times']['predictor_time_seconds']
    accumulated_metrics['peak_memory'] = max(
        accumulated_metrics['peak_memory'],
        performance_metrics['memory_usage']['peak_memory_bytes']
    )
    accumulated_metrics['total_gen_memory'] += performance_metrics['memory_usage']['generator_memory_bytes']
    accumulated_metrics['total_pred_memory'] += performance_metrics['memory_usage']['predictor_memory_bytes']
    
    # 计算平均性能指标
    avg_metrics = {
        'avg_fps': accumulated_metrics['total_frames'] / accumulated_metrics['total_time'] if accumulated_metrics['total_time'] > 0 else 0,
        'avg_gen_time': accumulated_metrics['total_gen_time'] / accumulated_metrics['total_frames'],
        'avg_pred_time': accumulated_metrics['total_pred_time'] / accumulated_metrics['total_frames'],
        'avg_gen_memory': accumulated_metrics['total_gen_memory'] / accumulated_metrics['total_frames'],
        'avg_pred_memory': accumulated_metrics['total_pred_memory'] / accumulated_metrics['total_frames'],
        'peak_memory': accumulated_metrics['peak_memory']
    }
    
    # 转换内存单位为MB
    current_metrics = {
        'fps': performance_metrics['fps'],
        'gen_time': performance_metrics['component_times']['generator_time_seconds'],
        'pred_time': performance_metrics['component_times']['predictor_time_seconds'],
        'gen_memory': performance_metrics['memory_usage']['generator_memory_bytes'] / 1024**2,
        'pred_memory': performance_metrics['memory_usage']['predictor_memory_bytes'] / 1024**2,
        'peak_memory': performance_metrics['memory_usage']['peak_memory_bytes'] / 1024**2
    }
    
    # 转换平均指标中的内存单位为MB
    avg_metrics['avg_gen_memory'] /= 1024**2
    avg_metrics['avg_pred_memory'] /= 1024**2
    avg_metrics['peak_memory'] /= 1024**2
    
    return current_metrics, accumulated_metrics, avg_metrics


def log_performance_metrics(current_metrics, avg_metrics, logger=None):
    """
    记录性能指标到日志和控制台
    
    参数:
        current_metrics: 当前帧指标字典
        avg_metrics: 平均指标字典
        logger: 日志记录器对象(可选)
    """
    # 打印性能指标
    print("\n=== 当前帧性能指标 ===")
    print(f"FPS: {current_metrics['fps']:.2f}")
    print(f"生成器时间: {current_metrics['gen_time']:.4f}s")
    print(f"预测器时间: {current_metrics['pred_time']:.4f}s")
    print(f"生成器内存: {current_metrics['gen_memory']:.2f}MB")
    print(f"预测器内存: {current_metrics['pred_memory']:.2f}MB")
    print(f"峰值内存: {current_metrics['peak_memory']:.2f}MB")
    
    print("\n=== 平均性能指标 ===")
    print(f"平均FPS: {avg_metrics['avg_fps']:.2f}")
    print(f"生成器平均时间: {avg_metrics['avg_gen_time']:.4f}s")
    print(f"预测器平均时间: {avg_metrics['avg_pred_time']:.4f}s")
    print(f"生成器平均内存: {avg_metrics['avg_gen_memory']:.2f}MB")
    print(f"预测器平均内存: {avg_metrics['avg_pred_memory']:.2f}MB")
    print(f"峰值内存: {avg_metrics['peak_memory']:.2f}MB")
    
    # 记录到日志
    if logger:
        logger.info("\n=== 当前帧性能指标 ===")
        logger.info(f"FPS: {current_metrics['fps']:.2f}")
        logger.info(f"生成器时间: {current_metrics['gen_time']:.4f}s")
        logger.info(f"预测器时间: {current_metrics['pred_time']:.4f}s")
        logger.info(f"生成器内存: {current_metrics['gen_memory']:.2f}MB")
        logger.info(f"预测器内存: {current_metrics['pred_memory']:.2f}MB")
        logger.info(f"峰值内存: {current_metrics['peak_memory']:.2f}MB")
        
        logger.info("\n=== 平均性能指标 ===")
        logger.info(f"平均FPS: {avg_metrics['avg_fps']:.2f}")
        logger.info(f"生成器平均时间: {avg_metrics['avg_gen_time']:.4f}s")
        logger.info(f"预测器平均时间: {avg_metrics['avg_pred_time']:.4f}s")
        logger.info(f"生成器平均内存: {avg_metrics['avg_gen_memory']:.2f}MB")
        logger.info(f"预测器平均内存: {avg_metrics['avg_pred_memory']:.2f}MB")
        logger.info(f"峰值内存: {avg_metrics['peak_memory']:.2f}MB")