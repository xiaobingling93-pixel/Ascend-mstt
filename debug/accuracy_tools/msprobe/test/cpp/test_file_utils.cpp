#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "test_utils.hpp"
#include "utils/FileUtils.h"

using namespace MindStudioDebugger;
using namespace MindStudioDebugger::FileUtils;

namespace MsProbeTest {

class FileUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建目录
        ASSERT_EQ(mkdir(testDir.c_str(), 0750), 0);
        ASSERT_EQ(mkdir(testDirSub.c_str(), 0750), 0);
        // 创建文件
        std::ofstream file(testRegularFile);
        file.close();
        // 创建符号链接
        ASSERT_EQ(symlink(GetAbsPath(testRegularFile).c_str(), testLink.c_str()), 0);
        ASSERT_EQ(mkfifo(testFifo.c_str(), 0640), 0);
    }

    void TearDown() override {
        // 删除测试目录和文件
        TEST_ExecShellCommand("rm -rf " + testDir);
    }

    const std::string testDir = "./FileUtilsTest";
    const std::string testDirSub = testDir + "/subdir";
    const std::string testRegularFile = testDir + "/RegularFile.txt";
    const std::string testNotExistsFile = testDir + "/NotExistsFile.txt";
    const std::string testLink = testDir + "/testlink";
    const std::string testFifo = testDir + "/testfifo";
};

TEST_F(FileUtilsTest, TestIsPathExist)
{
    EXPECT_TRUE(IsPathExist("/"));
    EXPECT_TRUE(IsPathExist("."));
    EXPECT_TRUE(IsPathExist(testRegularFile));
    EXPECT_FALSE(IsPathExist(testNotExistsFile));
}

TEST_F(FileUtilsTest, TestGetAbsPath)
{
    std::string pwd = Trim(TEST_ExecShellCommand("pwd"));
    EXPECT_EQ(pwd, GetAbsPath("."));
    EXPECT_EQ(pwd + "/testpath", GetAbsPath("./testpath"));
    EXPECT_EQ(pwd + "/testpath", GetAbsPath("./testpath/"));
    EXPECT_EQ(pwd + "/testpath", GetAbsPath("./subdir/../testpath"));
    EXPECT_EQ(pwd + "/testpath", GetAbsPath("subdir/subdir/.././../testpath"));
    EXPECT_EQ(pwd + "/subdir/testpath", GetAbsPath("./subdir/.././/subdir/testpath"));
}

TEST_F(FileUtilsTest, TestIsDir)
{
    EXPECT_TRUE(IsDir("/"));
    EXPECT_TRUE(IsDir("./"));
    EXPECT_TRUE(IsDir(testDirSub));
    EXPECT_FALSE(IsDir(testRegularFile));
    EXPECT_FALSE(IsDir(testFifo));
}

TEST_F(FileUtilsTest, TestIsRegularFile)
{
    EXPECT_TRUE(IsRegularFile(testRegularFile));
    EXPECT_FALSE(IsRegularFile(testDirSub));
    EXPECT_TRUE(IsRegularFile(testLink));
    EXPECT_FALSE(IsRegularFile(testFifo));
    EXPECT_FALSE(IsRegularFile(testNotExistsFile));
}

TEST_F(FileUtilsTest, TestIsFileSymbolLink)
{
    EXPECT_TRUE(IsFileSymbolLink(testLink));
    EXPECT_FALSE(IsFileSymbolLink(testDirSub));
    EXPECT_FALSE(IsFileSymbolLink(testNotExistsFile));
    EXPECT_FALSE(IsFileSymbolLink(testRegularFile));
    EXPECT_FALSE(IsFileSymbolLink(testFifo));
}

TEST_F(FileUtilsTest, TestIsPathCharactersValid)
{
    std::string validPath = "/tmp/FileUtilsTest/testfile.txt";
    std::string invalidPath1 = "/tmp/FileUtilsTest/<>:|?*\"";
    std::string invalidPath2 = " /tmp/FileUtilsTest/testfile.txt";
    EXPECT_TRUE(IsPathCharactersValid("123456789"));
    EXPECT_TRUE(IsPathCharactersValid(validPath));
    EXPECT_FALSE(IsPathCharactersValid(""));
    EXPECT_FALSE(IsPathCharactersValid(invalidPath1));
    EXPECT_FALSE(IsPathCharactersValid(invalidPath2));
}

TEST_F(FileUtilsTest, TestIsFileReadable)
{
    TEST_ExecShellCommand("chmod -r " + testRegularFile);
    EXPECT_FALSE(IsFileReadable(testRegularFile));
    TEST_ExecShellCommand("chmod +r " + testRegularFile);
    EXPECT_TRUE(IsFileReadable(testRegularFile));
    TEST_ExecShellCommand("chmod -r " + testDirSub);
    EXPECT_FALSE(IsFileReadable(testDirSub));
    TEST_ExecShellCommand("chmod +r " + testDirSub);
    EXPECT_TRUE(IsFileReadable(testDirSub));
}

TEST_F(FileUtilsTest, TestIsFileWritable)
{
    TEST_ExecShellCommand("chmod -w " + testRegularFile);
    EXPECT_FALSE(IsFileWritable(testRegularFile));
    TEST_ExecShellCommand("chmod +w " + testRegularFile);
    EXPECT_TRUE(IsFileWritable(testRegularFile));
    TEST_ExecShellCommand("chmod -w " + testDirSub);
    EXPECT_FALSE(IsFileWritable(testDirSub));
    TEST_ExecShellCommand("chmod +w " + testDirSub);
    EXPECT_TRUE(IsFileWritable(testDirSub));
}

TEST_F(FileUtilsTest, TestIsFileExecutable)
{
    TEST_ExecShellCommand("chmod -x " + testRegularFile);
    EXPECT_FALSE(IsFileExecutable(testRegularFile));
    TEST_ExecShellCommand("chmod +x " + testRegularFile);
    EXPECT_TRUE(IsFileExecutable(testRegularFile));
    TEST_ExecShellCommand("chmod -x " + testDirSub);
    EXPECT_FALSE(IsFileExecutable(testDirSub));
    TEST_ExecShellCommand("chmod +x " + testDirSub);
    EXPECT_TRUE(IsFileExecutable(testDirSub));
}

TEST_F(FileUtilsTest, TestIsDirReadable)
{
    EXPECT_TRUE(".");
    EXPECT_TRUE(IsDirReadable(testDirSub));
    TEST_ExecShellCommand("chmod 100 " + testDirSub);
    EXPECT_FALSE(IsDirReadable(testDirSub));
    TEST_ExecShellCommand("chmod 400 " + testDirSub);
    EXPECT_FALSE(IsDirReadable(testDirSub));
    TEST_ExecShellCommand("chmod 500 " + testDirSub);
    EXPECT_TRUE(IsDirReadable(testDirSub));
}

TEST_F(FileUtilsTest, TestGetParentDir)
{
    EXPECT_EQ("/tmp/FileUtilsTest", GetParentDir("/tmp/FileUtilsTest/dir"));
    EXPECT_EQ("/tmp/FileUtilsTest", GetParentDir("/tmp/FileUtilsTest/"));
    EXPECT_EQ("./FileUtilsTest", GetParentDir("./FileUtilsTest/testfile.txt"));
    EXPECT_EQ(".", GetParentDir("testfile.txt"));
    EXPECT_EQ(".", GetParentDir(""));
}

TEST_F(FileUtilsTest, TestGetFileName)
{
    EXPECT_EQ("dir", GetFileName("/tmp/FileUtilsTest/dir"));
    EXPECT_EQ("", GetFileName("/tmp/FileUtilsTest/"));
    EXPECT_EQ("testfile.txt", GetFileName("./FileUtilsTest/testfile.txt"));
    EXPECT_EQ("testfile.txt", GetFileName("testfile.txt"));
    EXPECT_EQ("", GetFileName(""));
}

TEST_F(FileUtilsTest, TestGetFileBaseName)
{
    EXPECT_EQ("dir", GetFileBaseName("/tmp/FileUtilsTest/dir"));
    EXPECT_EQ("", GetFileBaseName("/tmp/FileUtilsTest/"));
    EXPECT_EQ("testfile", GetFileBaseName("./FileUtilsTest/testfile.txt"));
    EXPECT_EQ("testfile", GetFileBaseName("testfile.txt"));
    EXPECT_EQ("testfile", GetFileBaseName("testfile"));
}

TEST_F(FileUtilsTest, TestGetFileSuffix)
{
    EXPECT_EQ("", GetFileSuffix("/tmp/FileUtilsTest/dir"));
    EXPECT_EQ("", GetFileSuffix("/tmp/FileUtilsTest/"));
    EXPECT_EQ("txt", GetFileSuffix("./FileUtilsTest/testfile.txt"));
    EXPECT_EQ("txt", GetFileSuffix("testfile.txt"));
    EXPECT_EQ("", GetFileSuffix("testfile"));
    EXPECT_EQ("", GetFileSuffix("testfile."));
}

TEST_F(FileUtilsTest, TestCheckFileRWX)
{
    TEST_ExecShellCommand("chmod 640 " + testRegularFile);
    EXPECT_TRUE(CheckFileRWX(testRegularFile, "rw"));
    EXPECT_FALSE(CheckFileRWX(testRegularFile, "rx"));
    TEST_ExecShellCommand("chmod 750 " + testDirSub);
    EXPECT_TRUE(CheckFileRWX(testDirSub, "rwx"));
}

TEST_F(FileUtilsTest, TestIsPathLengthLegal)
{
    std::string maxFile = std::string(FILE_NAME_LENGTH_MAX, 'a');
    std::string longFile = std::string(FILE_NAME_LENGTH_MAX + 1, 'a');
    std::string maxPath(FULL_PATH_LENGTH_MAX, '/');
    std::string longPath = maxPath + "/";
    EXPECT_TRUE(IsPathLengthLegal(maxFile));
    EXPECT_TRUE(IsPathLengthLegal(maxPath));
    EXPECT_FALSE(IsPathLengthLegal(longFile));
    EXPECT_FALSE(IsPathLengthLegal(longPath));
    EXPECT_FALSE(IsPathLengthLegal(""));
}

TEST_F(FileUtilsTest, TestIsPathDepthValid)
{
    EXPECT_TRUE(IsPathDepthValid(""));
    EXPECT_TRUE(IsPathDepthValid(std::string(PATH_DEPTH_MAX, PATH_SEPARATOR)));
    EXPECT_FALSE(IsPathDepthValid(std::string(PATH_DEPTH_MAX + 1, PATH_SEPARATOR)));
}

TEST_F(FileUtilsTest, TestIsFileOwner)
{
    EXPECT_TRUE(IsFileOwner(testRegularFile));
    EXPECT_TRUE(IsFileOwner(testDirSub));
    EXPECT_FALSE(IsFileOwner("/"));
}

TEST_F(FileUtilsTest, TestDeleteFile)
{
    ASSERT_TRUE(IsPathExist(testRegularFile));
    EXPECT_EQ(DeleteFile(testLink), DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK);
    EXPECT_EQ(DeleteFile(testRegularFile), DebuggerErrno::OK);
    EXPECT_FALSE(IsPathExist(testRegularFile));
    EXPECT_EQ(DeleteFile(testRegularFile), DebuggerErrno::OK);
    EXPECT_EQ(DeleteFile(testFifo), DebuggerErrno::OK);
    EXPECT_EQ(DeleteFile(testDirSub), DebuggerErrno::OK);
    EXPECT_EQ(DeleteFile(testDir), DebuggerErrno::ERROR_SYSCALL_FAILED);
    EXPECT_EQ(DeleteFile(testLink), DebuggerErrno::OK);
}

TEST_F(FileUtilsTest, TestDeleteDir)
{
    ASSERT_TRUE(IsPathExist(testDirSub));
    EXPECT_EQ(DeleteDir(testDirSub), DebuggerErrno::OK);
    EXPECT_FALSE(IsPathExist(testDirSub));
    EXPECT_EQ(DeleteDir(testDirSub), DebuggerErrno::OK);
    std::string subSubDir = testDirSub + "/subdir";
    std::string subSubFile = testDirSub + "/subfile";
    TEST_ExecShellCommand("mkdir " + testDirSub);
    TEST_ExecShellCommand("mkdir " + subSubDir);
    TEST_ExecShellCommand("touch " + subSubFile);
    EXPECT_EQ(DeleteDir(testLink), DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK);
    EXPECT_EQ(DeleteDir(testRegularFile), DebuggerErrno::ERROR_SYSCALL_FAILED);
    EXPECT_EQ(DeleteDir(testDirSub), DebuggerErrno::ERROR_SYSCALL_FAILED);
    EXPECT_EQ(DeleteDir(testDirSub, true), DebuggerErrno::OK);
    EXPECT_FALSE(IsPathExist(testDirSub));
}

TEST_F(FileUtilsTest, TestCreateDir)
{
    ASSERT_TRUE(IsPathExist(testDirSub));
    EXPECT_EQ(CreateDir(testDirSub), DebuggerErrno::OK);
    TEST_ExecShellCommand("rm -rf " + testDirSub);
    ASSERT_FALSE(IsPathExist(testDirSub));
    EXPECT_EQ(CreateDir(testDirSub), DebuggerErrno::OK);
    EXPECT_TRUE(IsPathExist(testDirSub));
    TEST_ExecShellCommand("rm -rf " + testDirSub);
    std::string subSubDir = testDirSub + "/subdir";
    EXPECT_EQ(CreateDir(subSubDir), DebuggerErrno::ERROR_DIR_NOT_EXISTS);
    EXPECT_EQ(CreateDir(subSubDir, true), DebuggerErrno::OK);
    EXPECT_TRUE(IsPathExist(subSubDir));
    EXPECT_TRUE(CheckFileRWX(subSubDir, "rwx"));
    TEST_ExecShellCommand("rm -rf " + testDirSub);
    EXPECT_EQ(CreateDir(subSubDir, true, 0750), DebuggerErrno::OK);
    EXPECT_TRUE(CheckFileRWX(testDirSub, "rwx"));
    EXPECT_TRUE(CheckFileRWX(subSubDir, "rwx"));
}

TEST_F(FileUtilsTest, TestChmod)
{
    EXPECT_EQ(Chmod(testNotExistsFile, 0640), DebuggerErrno::ERROR_FILE_NOT_EXISTS);
    EXPECT_EQ(Chmod(testRegularFile, 0440), DebuggerErrno::OK);
    EXPECT_FALSE(IsFileWritable(testRegularFile));
    EXPECT_EQ(Chmod(testDirSub, 0550), DebuggerErrno::OK);
    EXPECT_FALSE(IsFileWritable(testDirSub));
    EXPECT_EQ(Chmod(testRegularFile, 0640), DebuggerErrno::OK);
    EXPECT_TRUE(IsFileWritable(testRegularFile));
    EXPECT_EQ(Chmod(testLink, 0640), DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK);
    EXPECT_EQ(Chmod("", 0640), DebuggerErrno::ERROR_FILE_NOT_EXISTS);
    EXPECT_EQ(Chmod("/", 0750), DebuggerErrno::ERROR_SYSCALL_FAILED);
}

TEST_F(FileUtilsTest, TestGetFileSize)
{
    size_t size;
    EXPECT_EQ(GetFileSize(testRegularFile, size), DebuggerErrno::OK);
    EXPECT_EQ(size, 0);
    TEST_ExecShellCommand("echo \"123456789\" > " + testRegularFile);
    EXPECT_EQ(GetFileSize(testRegularFile, size), DebuggerErrno::OK);
    EXPECT_EQ(size, 10);
    EXPECT_EQ(GetFileSize(testNotExistsFile, size), DebuggerErrno::ERROR_FILE_NOT_EXISTS);
    EXPECT_EQ(GetFileSize(testDirSub, size), DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE);
    EXPECT_EQ(GetFileSize(testFifo, size), DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE);
}

TEST_F(FileUtilsTest, TestOpenFileRead)
{
    std::ifstream ifs;
    EXPECT_EQ(OpenFile(testNotExistsFile, ifs), DebuggerErrno::ERROR_FILE_NOT_EXISTS);
    TEST_ExecShellCommand("chmod -r " + testRegularFile);
    EXPECT_EQ(OpenFile(testRegularFile, ifs), DebuggerErrno::ERROR_PERMISSION_DENINED);
    TEST_ExecShellCommand("chmod +r " + testRegularFile);
    EXPECT_EQ(OpenFile(testLink, ifs), DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK);
    TEST_ExecShellCommand("echo \"123456789\" > " + testRegularFile);
    ASSERT_EQ(OpenFile(testRegularFile, ifs), DebuggerErrno::OK);
    ASSERT_TRUE(ifs.is_open());
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_EQ(content, "123456789\n");
    ifs.close();
}

TEST_F(FileUtilsTest, TestOpenFileWrite)
{
    std::ofstream ofs;
    ASSERT_EQ(OpenFile(testRegularFile, ofs), DebuggerErrno::OK);
    ofs << "123456789";
    ofs.close();
    std::ifstream ifs(testRegularFile, std::ios::in);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    EXPECT_EQ(content, "123456789");
}

TEST_F(FileUtilsTest, TestCheckFileSuffixAndSize)
{
    EXPECT_EQ(CheckFileSuffixAndSize(testRegularFile, FileType::COMMON), DebuggerErrno::OK);
    EXPECT_EQ(CheckFileSuffixAndSize(testRegularFile, FileType::JSON), DebuggerErrno::ERROR_UNKNOWN_FILE_SUFFIX);
    std::string sparseKpl = testDir + "/test.kpl";
    std::string sparseNpy = testDir + "/test.npy";
    std::string sparseJson = testDir + "/test.json";
    std::string sparsePt = testDir + "/test.pt";
    std::string sparseCsv = testDir + "/test.csv";
    std::string sparseYaml = testDir + "/test.yaml";
    TEST_ExecShellCommand("truncate -s 1G " + sparseCsv);
    EXPECT_EQ(CheckFileSuffixAndSize(sparseCsv, FileType::CSV), DebuggerErrno::OK);
    TEST_ExecShellCommand("rm " + sparseCsv);
    TEST_ExecShellCommand("truncate -s 1025M " + sparseCsv);
    EXPECT_EQ(CheckFileSuffixAndSize(sparseCsv, FileType::CSV), DebuggerErrno::ERROR_FILE_TOO_LARGE);
    TEST_ExecShellCommand("truncate -s 1025M " + sparseKpl);
    EXPECT_EQ(CheckFileSuffixAndSize(sparseKpl, FileType::PKL), DebuggerErrno::ERROR_FILE_TOO_LARGE);
    TEST_ExecShellCommand("truncate -s 11G " + sparseNpy);
    EXPECT_EQ(CheckFileSuffixAndSize(sparseNpy, FileType::NUMPY), DebuggerErrno::ERROR_FILE_TOO_LARGE);
    TEST_ExecShellCommand("truncate -s 1025M " + sparseJson);
    EXPECT_EQ(CheckFileSuffixAndSize(sparseJson, FileType::JSON), DebuggerErrno::ERROR_FILE_TOO_LARGE);
    TEST_ExecShellCommand("truncate -s 11G " + sparsePt);
    EXPECT_EQ(CheckFileSuffixAndSize(sparsePt, FileType::PT), DebuggerErrno::ERROR_FILE_TOO_LARGE);
    TEST_ExecShellCommand("truncate -s 10241K " + sparseYaml);
    EXPECT_EQ(CheckFileSuffixAndSize(sparseYaml, FileType::YAML), DebuggerErrno::ERROR_FILE_TOO_LARGE);
}

TEST_F(FileUtilsTest, TestCheckDirCommon)
{
    EXPECT_EQ(CheckDirCommon(""), DebuggerErrno::ERROR_CANNOT_PARSE_PATH);
    EXPECT_EQ(CheckDirCommon(testNotExistsFile), DebuggerErrno::ERROR_FILE_NOT_EXISTS);
    EXPECT_EQ(CheckDirCommon(testRegularFile), DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE);
    std::string linkdir = testDir + "/linkdir";
    TEST_ExecShellCommand("ln -s " + GetAbsPath(testDirSub) + " " + linkdir);
    EXPECT_EQ(CheckDirCommon(linkdir), DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK);
    EXPECT_EQ(CheckDirCommon(testDirSub), DebuggerErrno::OK);
    TEST_ExecShellCommand("chmod -r " + testDirSub);
    EXPECT_EQ(CheckDirCommon(testDirSub), DebuggerErrno::ERROR_PERMISSION_DENINED);
}

TEST_F(FileUtilsTest, TestCheckFileBeforeRead)
{
    EXPECT_EQ(CheckFileBeforeRead(""), DebuggerErrno::ERROR_CANNOT_PARSE_PATH);
    EXPECT_EQ(CheckFileBeforeRead(testNotExistsFile), DebuggerErrno::ERROR_FILE_NOT_EXISTS);
    EXPECT_EQ(CheckFileBeforeRead(testLink), DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK);
    EXPECT_EQ(CheckFileBeforeRead(testRegularFile), DebuggerErrno::OK);
    TEST_ExecShellCommand("chmod -r " + testRegularFile);
    EXPECT_EQ(CheckFileBeforeRead(testRegularFile), DebuggerErrno::ERROR_PERMISSION_DENINED);
}

TEST_F(FileUtilsTest, TestCheckFileBeforeCreateOrWrite)
{
    EXPECT_EQ(CheckFileBeforeCreateOrWrite(""), DebuggerErrno::ERROR_CANNOT_PARSE_PATH);
    EXPECT_EQ(CheckFileBeforeCreateOrWrite(testNotExistsFile), DebuggerErrno::OK);
    EXPECT_EQ(CheckFileBeforeCreateOrWrite(testRegularFile), DebuggerErrno::ERROR_FILE_ALREADY_EXISTS);
    EXPECT_EQ(CheckFileBeforeCreateOrWrite(testRegularFile, true), DebuggerErrno::OK);
    TEST_ExecShellCommand("chmod -w " + testRegularFile);
    EXPECT_EQ(CheckFileBeforeCreateOrWrite(testRegularFile, true), DebuggerErrno::ERROR_PERMISSION_DENINED);
    EXPECT_EQ(CheckFileBeforeCreateOrWrite("/", true), DebuggerErrno::ERROR_PERMISSION_DENINED);
}

}
