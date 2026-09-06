#include <windows.h>
#include <winternl.h>
#include <stddef.h>
#include <stdio.h>
int main(void) {
    printf("{\"handle\":%zu,\"wchar\":%zu,\"by_handle_info\":%zu,\"disposition\":%zu,\"disposition_ex\":%zu,\"io_status\":%zu,\"io_information_offset\":%zu,\"rename_root_offset\":%zu,\"rename_length_offset\":%zu,\"rename_name_offset\":%zu}\n",
           sizeof(HANDLE), sizeof(WCHAR), sizeof(BY_HANDLE_FILE_INFORMATION),
           sizeof(FILE_DISPOSITION_INFO), sizeof(FILE_DISPOSITION_INFO_EX),
           sizeof(IO_STATUS_BLOCK), offsetof(IO_STATUS_BLOCK, Information),
           offsetof(FILE_RENAME_INFO, RootDirectory), offsetof(FILE_RENAME_INFO, FileNameLength),
           offsetof(FILE_RENAME_INFO, FileName));
    return 0;
}
