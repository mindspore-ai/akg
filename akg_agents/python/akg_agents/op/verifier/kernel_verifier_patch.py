    def _pack_directory(self, dir_path: str) -> bytes:
        """将目录打包为zip字节流"""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dir_path)
                    zip_file.write(file_path, arcname)
        return zip_buffer.getvalue()

    async def run_verify(self, verify_dir: str, timeout: int = 300):
        """
        运行验证脚本
        """
        # 打包验证目录
        package_data = self._pack_directory(verify_dir)
        
        # 调用Worker执行验证
        logger.info(f"[{self.op_name}] 调用Worker执行验证, timeout={timeout}秒")
        success, log = await self.worker.verify(package_data, self.task_id, self.op_name, timeout)
        
        if success:
            logger.info(f"[{self.op_name}] 验证执行成功")
        else:
            logger.error(f"[{self.op_name}] 验证执行失败")
            
        return success, log

